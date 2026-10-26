"""Bounded, train-only serving diagnostics; no numerical gate is relaxed."""
import hashlib
import importlib
import math
import re

from .research_schema import digest

VERSION = "research-lora-serving-diagnostic-v1.1"
SEED = 20261011
TOLERANCE = 1e-4
ORDER = ("base", "selected", "selected", "alias", "selected", "base")
MAX_REQUESTS = 2 * 3 * len(ORDER)


def prefix_plan(records, tokenizer):
    from .research_sft import build_expert, tokenize_expert
    if not records or any(r.get("split") != "train" for r in records):
        raise ValueError("only registered train records permitted")
    plan = []
    for depth in (1, 2, 3):
        row = next((r for r in records if r["reference_depth"] == depth), None)
        if row is None:
            raise ValueError("missing fixed train depth")
        tokens = tokenize_expert(build_expert(row), tokenizer, max_length=4096)
        ids = tokens["input_ids"][:tokens["assistant_token_ranges"][-1][1] - 1]
        if not ids or len(ids) >= 4096:
            raise ValueError("invalid teacher-forced prefix")
        plan.append({"sample_id": row["sample_id"], "depth": depth,
                     "prompt_token_ids": ids, "prompt_sha256": digest(ids)})
    return plan


def distribution(reply, prompt_length, vocab_size):
    """API token_id keys prevent decoded-text collisions; require FULL coverage."""
    usage = reply.get("usage")
    if not usage or usage.get("prompt_tokens") != prompt_length or usage.get("completion_tokens") != 1:
        raise ValueError("usage/token contract mismatch")
    choices = reply.get("choices", [])
    if len(choices) != 1:
        raise ValueError("exactly one completion required")
    lp = choices[0].get("logprobs", {})
    tokens = lp.get("tokens", [])
    if len(tokens) != 1 or not re.fullmatch(r"token_id:\d+", tokens[0]):
        raise ValueError("numeric generated token ID required")
    try:
        generated = int(tokens[0].split(":")[1])
    except ValueError as exc:
        raise ValueError("invalid generated token ID") from exc
    scores = lp.get("top_logprobs", [])
    if len(scores) != 1 or not isinstance(scores[0], dict):
        raise ValueError("one full distribution required")
    parsed = {}
    for key, value in scores[0].items():
        if not re.fullmatch(r"token_id:\d+", key) or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid/nonfinite token logprob")
        try:
            token = int(key.split(":")[1])
            score = float(value)
        except ValueError as exc:
            raise ValueError("invalid token ID/logprob") from exc
        if token in parsed or value > 1e-6:
            raise ValueError("duplicate ID/positive logprob")
        parsed[token] = score
    if set(parsed) != set(range(vocab_size)) or generated not in parsed:
        raise ValueError("incomplete vocabulary coverage; never intersect away missing tokens")
    generated_lp = lp.get("token_logprobs", [])
    if len(generated_lp) != 1 or generated_lp[0] != parsed[generated]:
        raise ValueError("generated logprob differs from vocabulary score")
    # Full distribution is expected to be normalized, not raw logits.
    peak = max(parsed.values())
    log_z = peak + math.log(math.fsum(math.exp(x - peak) for x in parsed.values()))
    if abs(log_z) > 1e-4:
        raise ValueError("distribution not normalized")
    return generated, [parsed[i] for i in range(vocab_size)]


def compare(left, right):
    if left["prompt_sha256"] != right["prompt_sha256"] or len(left["logprobs"]) != len(right["logprobs"]):
        raise ValueError("probe identity/coverage mismatch")
    differences = [abs(a - b) for a, b in zip(left["logprobs"], right["logprobs"], strict=True)]
    if not differences or not all(math.isfinite(x) for x in differences):
        raise ValueError("empty/nonfinite comparison")
    targets = sorted(range(len(differences)), key=lambda i: left["logprobs"][i], reverse=True)[:20]
    delta = max(differences)
    return {"full_vocab_max_delta": delta, "worst_token_id": differences.index(delta),
            "left_top20_fixed_target_ids": targets,
            "left_top20_target_max_delta": max(differences[i] for i in targets),
            "greedy_equal": left["generated_id"] == right["generated_id"],
            "unchanged_gate_1e_4_passed": delta <= TOLERANCE and left["generated_id"] == right["generated_id"]}


def tensor_receipt(tensor):
    torch = importlib.import_module("torch")
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def audit_resident(manager, source, *, torch, device_type="cuda"):
    """vLLM 0.17.0 Qwen2 TP=1: audit actual CUDA slots, not PEFT trainer storage.

    Known transforms only: packed q/k/v or gate/up, dtype cast, alpha/r folded
    into B. Require exact transformed equality, complete source coverage and
    zero padding. Unsupported layouts fail closed, not an inferred success.
    """
    receipts = []
    slots = manager.lora_index_to_id
    if not any(x is not None for x in slots):
        raise ValueError("no active GPU LoRA slot")
    for slot, adapter_id in enumerate(slots):
        if adapter_id is None:
            continue
        model = manager._registered_adapters[adapter_id]
        seen = set()
        rows = []
        inactive = []
        for module_name, module in manager.modules.items():
            weights = manager._get_lora_layer_weights(model, module_name)
            leaf = module_name.rsplit(".", 1)[-1]
            if weights is None:
                # vLLM wraps supported embedding/head modules even when this
                # seven-projection adapter has no weights for them. Verify reset.
                if leaf not in {"embed_tokens", "lm_head"}:
                    raise ValueError("missing Qwen2 LoRA projection: " + module_name)
                for label in ("A", "B"):
                    stacked = getattr(module, "lora_" + label.lower() + "_stacked")
                    stacks = stacked if isinstance(stacked, (list, tuple)) else (stacked,)
                    for tensor in stacks:
                        live = tensor[slot]
                        if live.device.type != device_type:
                            raise ValueError("inactive slot not on expected device")
                        inactive.append({"module": module_name, "label": label, "actual_slot": tensor_receipt(live),
                                         "zero": torch.count_nonzero(live).item() == 0})
                continue
            packed = manager.packed_modules_mapping.get(leaf, [leaf])
            if isinstance(weights.lora_a, list):
                names = packed
                alphas = weights.lora_alphas
                scales = weights.scaling
            else:
                names, alphas, scales = [leaf], [weights.lora_alpha], [weights.scaling]
            if len(names) != len(module.lora_a_stacked) or len(names) != len(module.lora_b_stacked):
                raise ValueError("unsupported packed slot layout")
            if weights.rank != 8 or any(a != 16 for a in alphas) or any(s != 1 for s in scales):
                raise ValueError("unverified rank/alpha/scaling; require alpha/r=2 folded into B")
            for i, name in enumerate(names):
                unpacked = module_name.rsplit(".", 1)[0] + "." + name
                for label, stacked, multiplier in (("A", module.lora_a_stacked, 1), ("B", module.lora_b_stacked, 2)):
                    key = "base_model.model." + unpacked + ".lora_" + label + ".weight"
                    if key not in source or key in seen:
                        raise ValueError("missing/duplicate source mapping: " + key)
                    seen.add(key)
                    live = stacked[i][slot, 0]
                    if live.device.type != device_type or live.dtype not in (torch.bfloat16, torch.float32):
                        raise ValueError("expected actual CUDA floating LoRA slot")
                    expected = source[key].to(device=live.device, dtype=live.dtype) * multiplier
                    if live.ndim != 2 or any(a < b for a, b in zip(live.shape, expected.shape, strict=True)):
                        raise ValueError("unsupported TP/shape")
                    padded = torch.zeros_like(live)
                    padded[:expected.shape[0], :expected.shape[1]] = expected
                    row = {"source_key": key, "module": module_name, "slice": i, "slot": slot,
                           "source": tensor_receipt(source[key]), "actual_GPU_slot": tensor_receipt(live),
                           "expected_transformed_slot": tensor_receipt(padded), "B_scale_folded": multiplier,
                           "equal_expected_including_zero_padding": torch.equal(live, padded)}
                    rows.append(row)
        if seen != set(source) or len(seen) != 336:
            raise ValueError("incomplete 336-tensor source coverage")
        receipts.append({"adapter_id": adapter_id, "slot": slot, "rows": rows, "inactive_embedding_head_slots": inactive,
                         "transformed_slots_equal_source": all(r["equal_expected_including_zero_padding"] for r in rows)
                         and all(r["zero"] for r in inactive)})
    return receipts


def resident_worker(adapter_path):
    """Executed read-only inside ART's owned vLLM worker via trusted-code RPC."""
    torch = importlib.import_module("torch")
    get_worker = importlib.import_module("art.vllm").get_worker
    load_file = importlib.import_module("safetensors.torch").load_file
    worker = get_worker()
    manager = worker.model_runner.lora_manager._adapter_manager
    source = load_file(adapter_path, device="cpu")
    return {"resident": audit_resident(manager, source, torch=torch),
            "registered_ids": sorted(manager._registered_adapters),
            "active_slots": list(manager.lora_index_to_id),
            "packed_modules_mapping": manager.packed_modules_mapping,
            # TorchVersion is a str subclass, but vLLM's secure MsgPack
            # encoder rejects it. Return an exact builtin str, never enable pickle.
            "torch_version": str(torch.__version__)}

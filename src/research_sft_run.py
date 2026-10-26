"""Short SFT contracts: token-weighted updates, safe resume, validation selection.

No pretrained loads on import. Resume uses safetensors + JSON, never pickle.
GRPO starts contain adapter artifacts only, not optimizer or inherited .art state.
"""

import asyncio
import copy
import importlib
import math
import random
import shutil
from pathlib import Path

from .research_eval_backend import PromptBudgetExceeded, TransformersCPUBackend
from .research_model_identity import artifact_inventory, file_hash, resolve_identity
from .research_schema import digest, load_json
from .research_sft import pad_batch
from .research_sft_pilot import tensor_fingerprint

CONFIG_PATH = "configs/research/sft_short_v1.json"
FIXED = {
    "base_revision": "7ae557604adf67be50417f59c2c2f167def9a775", "seed": 20261011,
    "max_length": 4096, "micro_batch_size": 3, "gradient_accumulation": 4,
    "learning_rate": 0.0001, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "dtype": "bfloat16", "attention": "eager", "validation_scenarios": 180,
    "selection_order": ["answer-correct-count-descending", "reference-dynamic-valid-count-descending", "checkpoint-step-ascending"],
    "sampling_order": "seed-plus-epoch-shuffle-all-1200-no-drop", "scheduler": "constant",
    "automatic_retries": 0, "test_allowed": False,
    "purpose": "resource-limited-B1-candidate-not-full-epoch-not-effect-guarantee",
}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(__import__("json").dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_config(config: dict, *, parent: dict | None = None, start_step: int = 0) -> None:
    if set(config) != set(FIXED) | {"version", "terminal_step", "candidate_steps"}:
        raise ValueError("unknown/missing short SFT configuration keys")
    for key, value in FIXED.items():
        if config[key] != value or type(config[key]) is not type(value):
            raise ValueError("immutable SFT contract differs: " + key)
    end, steps = config["terminal_step"], config["candidate_steps"]
    if (type(end) is not int or not 0 <= start_step < end or not isinstance(steps, list) or not steps
            or any(type(s) is not int or not start_step < s <= end for s in steps)
            or steps != sorted(set(steps)) or steps[-1] != end):
        raise ValueError("candidate steps must be distinct future steps and include terminal step")
    if parent is None:
        if config["version"] != "research-sft-short-v1" or end != 10 or steps != [5, 10] or start_step != 0:
            raise ValueError("first run must use the locked 10-step contract")
    elif (not isinstance(config["version"], str) or not config["version"].startswith("research-sft-short-")
          or config["version"] == parent["version"] or start_step > parent["terminal_step"]
          or any(config[k] != parent[k] for k in FIXED)):
        raise ValueError("resume requires a versioned extension and unchanged optimizer/data contract")


def batch_indices(count: int, step: int, config: dict) -> list[int]:
    size = config["micro_batch_size"] * config["gradient_accumulation"]
    if count != 1200 or count % size or type(step) is not int or step < 1:
        raise ValueError("fixed complete 1200-record train corpus required")
    epoch, within = divmod(step - 1, count // size)
    order = list(range(count))
    random.Random(config["seed"] + epoch).shuffle(order)
    return order[within * size:(within + 1) * size]


def lora_parameters(model) -> dict:
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not parameters or any("lora_" not in n for n in parameters) or not sum(p.numel() for p in parameters.values()):
        raise ValueError("nonempty LoRA-only trainable parameters required")
    return parameters


def optimizer_update(model, optimizer, rows: list[dict], config: dict, torch, *, pad_id: int, device: str) -> dict:
    """One effective batch. Weight micro-losses by targets, not examples/lengths."""
    if len(rows) != config["micro_batch_size"] * config["gradient_accumulation"]:
        raise ValueError("incomplete effective batch")
    parameters = lora_parameters(model)
    before = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    total = sum(sum(label != -100 for label in r["labels"][1:]) for r in rows)
    if not total:
        raise ValueError("empty shifted supervision")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    weighted, audited = 0.0, 0.0
    micro_counts = []
    for start in range(0, len(rows), config["micro_batch_size"]):
        group = rows[start:start + config["micro_batch_size"]]
        batch = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in pad_batch(group, pad_token_id=pad_id).items()}
        targets = batch["labels"][:, 1:]
        count = int((targets != -100).sum().item())
        if not count:
            raise ValueError("empty micro-batch targets")
        output = model(**batch, use_cache=False)
        loss = output.loss
        manual = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                                                   targets.reshape(-1), ignore_index=-100)
        if (not torch.isfinite(loss).item() or loss.item() <= 0
                or not torch.allclose(loss.float(), manual.float(), atol=1e-4, rtol=1e-4)):
            raise ValueError("nonfinite/empty/mask-inconsistent loss")
        (loss * (count / total)).backward()
        weighted += loss.item() * count / total
        audited += manual.item() * count / total
        micro_counts.append(count)
        del output, manual, loss, targets, batch
    gradients = [p.grad for p in parameters.values() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
        raise ValueError("missing/nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters.values(), config["max_grad_norm"])
    if not math.isfinite(norm.item()) or norm.item() <= 0:
        raise ValueError("zero/nonfinite grad norm")
    optimizer.step()
    changed = sum(not torch.equal(before[n], p.detach().cpu()) for n, p in parameters.items())
    if not changed:
        raise ValueError("optimizer made no update")
    return {"loss": weighted, "manual_masked_loss": audited, "shifted_supervised_tokens": total,
            "micro_target_counts": micro_counts, "grad_norm_before_clip": norm.item(),
            "changed_tensors_this_update": changed, "trainable_parameters": sum(p.numel() for p in parameters.values()),
            "adapter_tensor_sha256": tensor_fingerprint(parameters, torch)}


def save_resume(directory: Path, model, optimizer, torch, *, step: int, contract: dict, cuda: bool) -> dict:
    """Only tensors + primitive JSON; optimizer parameter names bind ordering."""
    directory.mkdir(exist_ok=False)
    tensors = {"rng.cpu": torch.get_rng_state().clone()}
    if cuda:
        tensors["rng.cuda"] = torch.cuda.get_rng_state(0).cpu()
    state = optimizer.state_dict()
    scalar_state = {}
    for index, values in state["state"].items():
        scalar_state[str(index)] = {}
        for key, value in values.items():
            if torch.is_tensor(value):
                tensors[f"optimizer.{index}.{key}"] = value.detach().cpu().contiguous().clone()
            elif type(value) in (int, float, bool):
                scalar_state[str(index)][key] = value
            else:
                raise ValueError("unsupported optimizer state")
    importlib.import_module("safetensors.torch").save_file(tensors, str(directory / "state.safetensors"))
    receipt = {"version": "research-sft-resume-v1", "step": step, "next_step": step + 1,
               "contract": contract, "parameter_names": list(lora_parameters(model)),
               "adapter_tensor_sha256": tensor_fingerprint(lora_parameters(model), torch),
               "param_groups": state["param_groups"], "scalar_state": scalar_state, "cuda": cuda,
               "state_sha256": file_hash(directory / "state.safetensors")}
    write_json(directory / "resume.json", receipt)
    return receipt


def restore_resume(directory: Path, model, optimizer, torch, *, expected_contract: dict, device: str) -> dict:
    receipt = load_json(directory / "resume.json")
    if (receipt["version"] != "research-sft-resume-v1" or receipt["contract"] != expected_contract
            or receipt["parameter_names"] != list(lora_parameters(model))
            or receipt["adapter_tensor_sha256"] != tensor_fingerprint(lora_parameters(model), torch)
            or file_hash(directory / "state.safetensors") != receipt["state_sha256"]
            or type(receipt["step"]) is not int or receipt["step"] < 1 or receipt["next_step"] != receipt["step"] + 1
            or receipt["cuda"] != device.startswith("cuda")):
        raise ValueError("resume data/optimizer/adapter/state identity mismatch")
    tensors = importlib.import_module("safetensors.torch").load_file(str(directory / "state.safetensors"))
    state = {int(index): dict(values) for index, values in receipt["scalar_state"].items()}
    for name, tensor in tensors.items():
        if name.startswith("optimizer."):
            _, index, key = name.split(".")
            state[int(index)][key] = tensor
    expected_ids = set(optimizer.state_dict()["param_groups"][0]["params"])
    if set(state) != expected_ids or any(set(values) != {"step", "exp_avg", "exp_avg_sq"} for values in state.values()):
        raise ValueError("incomplete AdamW resume state")
    optimizer.load_state_dict({"state": state, "param_groups": receipt["param_groups"]})
    torch.set_rng_state(tensors["rng.cpu"])
    if receipt["cuda"]:
        torch.cuda.set_rng_state(tensors["rng.cuda"], 0)
    return receipt


def selection_key(row: dict) -> tuple:
    return (-row["answer_correct"], -row["dynamic_valid"], row["step"])


def select_checkpoint(candidates: list[dict], expected_steps: list[int]) -> dict:
    if sorted(row["step"] for row in candidates) != expected_steps:
        raise ValueError("missing/duplicate/unplanned candidate: no partial selection")
    for row in candidates:
        if (type(row["step"]) is not int or not row["comparison_valid"] or row["scenarios"] != 180
                or any(type(row[k]) is not int or not 0 <= row[k] <= 180 for k in ("answer_correct", "dynamic_valid"))):
            raise ValueError("incomplete/infra-invalid validation cannot select B1")
    return min(candidates, key=selection_key)


def stage_grpo_starts(best_spec: dict, output: Path, *, selection: dict) -> dict:
    """Prepare byte-independent G1/G2 adapters. No ART workspace/optimizer initialized."""
    identity = resolve_identity(best_spec)
    if best_spec["kind"] != "checkpoint" or selection["step"] != identity["checkpoint_step"]:
        raise ValueError("selected checkpoint required")
    output.mkdir(parents=True, exist_ok=False)
    starts = {}
    source = Path(identity["checkpoint_path"])
    for arm in ("G1", "G2"):
        workspace = output / arm
        adapter = workspace / "initial-adapter"
        adapter.mkdir(parents=True, exist_ok=False)
        for name in identity["checkpoint_files"]:
            shutil.copyfile(source / name, adapter / name)
        if artifact_inventory(adapter) != identity["checkpoint_files"]:
            raise ValueError("staged initial adapter bytes differ")
        spec = {**best_spec, "checkpoint_path": str(adapter.resolve())}
        copied = resolve_identity(spec)
        if copied["fingerprint"] != identity["fingerprint"]:
            raise ValueError("G1/G2 shared origin fingerprint mismatch")
        starts[arm] = {"model_spec": spec, "identity": copied, "optimizer_initialized": False,
                       "ART_workspace_initialized": False, "actual_GRPO_load_verified": False,
                       "must_create_fresh_optimizer": True, "step_origin": selection["step"]}
    receipt = {"version": "research-shared-sft-starts-v1", "selected": selection, "source": identity,
               "arms": starts, "GRPO_training_started": False, "R4_verified": False}
    write_json(output / "starts.json", receipt)
    return receipt


class SFTValidationBackend(TransformersCPUBackend):
    """Same Qwen codec/decoding protocol, explicit CUDA bf16 fresh PEFT reload.

    Not a CPU loader receipt; used only in the separate SFT selection runner.
    """

    def __init__(self, model, tokenizer, torch, identity: dict, versions: dict):
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self._template = tokenizer.chat_template
        self._lock, self._closed = asyncio.Lock(), False
        parameters = list(model.parameters())
        if (not parameters or any(str(p.device) != "cuda:0" for p in parameters)
                or list(model.active_adapters) != ["default"]):
            raise ValueError("validation requires explicit cuda:0/default adapter")
        self.receipt = {"backend": "transformers-cuda-sft-selection-v1", "real_model_loaded": True,
                        "identity": identity, "device": "cuda:0", "base_dtype": "bfloat16", "versions": versions,
                        "active_adapters": list(model.active_adapters), "source_weights_verified": True}

    def _generate(self, request: dict) -> dict:
        tokenizer, model, torch = self._tokenizer, self._model, self._torch
        if self._closed or model is None:
            raise RuntimeError("backend closed")
        inputs = tokenizer.apply_chat_template(copy.deepcopy(request["messages"]), tools=request["tools"],
                                              chat_template=self._template, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt").to("cuda:0")
        length = inputs["input_ids"].shape[-1]
        if length > request["max_prompt_tokens"]:
            raise PromptBudgetExceeded("prompt exceeds fixed token budget")
        if request["temperature"] != 0:
            raise ValueError("selection only supports locked deterministic decoding")
        config = importlib.import_module("transformers").GenerationConfig(
            max_new_tokens=request["max_new_tokens"], do_sample=False, num_beams=1, num_return_sequences=1,
            return_dict_in_generate=True, max_time=request["timeout_seconds"],
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(request["seed"])
            output = model.generate(**inputs, generation_config=config)
        ids = output.sequences[0, length:].tolist()
        stopped = bool(ids) and ids[-1] == tokenizer.eos_token_id
        text = tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False)
        return {"choices": [{"raw_text": text, "finish_reason": "stop" if stopped else "length"}],
                "usage": {"prompt_tokens": length, "completion_tokens": len(ids)}, "raw_token_ids": ids,
                "call_id_source": "adapter-generated-from-text-not-provider"}


def resume_contract(config: dict, receipt: dict, versions: dict) -> dict:
    return {"fixed_config_sha256": digest({k: config[k] for k in FIXED}),
            "data_and_tokenizer": receipt, "versions": versions,
            "optimizer": "AdamW-constant-token-weighted-no-scheduler-no-scaler"}

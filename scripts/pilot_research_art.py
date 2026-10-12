#!/usr/bin/env python3
"""Read-only ART 0.5.17/vLLM loading pilot for the already verified SFT step 2.

Default: identity/data preflight only. No optimizer/train/save/delete calls.
"""

import argparse
import asyncio
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend_lifecycle import install_monitor_workaround, managed_backend
from src.research_checker import protocol_plan
from src.research_environment import TOOL_SCHEMAS, ResearchEnvironment
from src.research_eval_backend import (
    ModelProtocolError,
    PromptBudgetExceeded,
    normalize_response,
)
from src.research_eval_protocol import plan_attempts
from src.research_evaluation import complete_ledger, evaluate_one
from src.research_model_identity import (
    artifact_inventory,
    file_hash,
    resolve_identity,
    verify_unchanged,
)
from src.research_registration import load_registered_development
from src.research_schema import digest, load_json, model_messages
from src.research_sft import build_expert, tokenize_expert

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_FILES = {
    "adapter_config.json": "204dd9ffa5f59ce420aadc4e9c2ec23a1bb998f8a27e98606fc3209e1fc3462f",
    "adapter_model.safetensors": "27d4b70288a82b3b2441b3b9ec53b754ec1580f5151a9d9814dfcd1286c3c210",
    "research_checkpoint.json": "eb56f59587da2f56a57d3ae1472e08201f129abc4b84d762c3258cf163c1c71f",
}
WALL_CAP = 900


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preflight(base: Path, adapter: Path):
    for name, sha in EXPECTED_FILES.items():
        if file_hash(adapter / name) != sha:
            raise ValueError("Drive source differs from verified SFT pilot: " + name)
    spec = {"kind": "checkpoint", "base_path": str(base), "base_revision": REVISION,
            "checkpoint_path": str(adapter), "checkpoint_step": 2}
    identity = resolve_identity(spec)
    if (identity["base_fingerprint"] != "5572b0e15dbd0cc2e6cc4f23aac1bab454184c1ad2170cd5bcd64230e535864c"
            or identity["checkpoint_fingerprint"] != "a726ec01c07a74e127edffe8ceb931f320877e6631fd5e849135c00e3d050216"):
        raise ValueError("Base/adapter inventory differs from verified SFT feedback")
    records, source = load_registered_development(ROOT, split="validation")
    selected = [next(r for r in records if r["reference_depth"] == d) for d in (1, 2, 3)]
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    protocol["modes"] = ["deterministic"]  # diagnostic subset, NOT formal preregistered comparison
    protocol["concurrency"] = 1
    return spec, identity, selected, source, protocol


async def run_art(output: Path, original: dict, records: list, protocol: dict, result: dict, begun: float, received: list):
    # ART pins torchao 0.15; PEFT 0.21 requires >=0.16. Unsloth requires PEFT >=0.18.
    for package, expected in (("openpipe-art", "0.5.17"), ("peft", "0.18.0"), ("torchao", "0.15.0")):
        if metadata.version(package) != expected:
            raise ValueError(f"ART pilot requires {package}=={expected}; repair dependencies before model loading")
    # Unsloth must precede Transformers/PEFT imports in this isolated backend environment.
    importlib.import_module("unsloth")
    torch = importlib.import_module("torch")
    art = importlib.import_module("art")
    peft = importlib.import_module("peft")
    safetensors = importlib.import_module("safetensors.torch")
    transformers = importlib.import_module("transformers")
    openai = importlib.import_module("openai")
    httpx = importlib.import_module("httpx")
    backend_class = importlib.import_module("art.local.backend").LocalBackend
    dirs = importlib.import_module("art.utils.output_dirs")
    if metadata.version("openpipe-art") != "0.5.17":
        raise ValueError("this private checkpoint-loading adapter is pinned to ART 0.5.17")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name or props.total_memory < 39 * 1024**3:
        raise ValueError("A100 40GB required")
    result["versions"] = {n: metadata.version(n) for n in ("openpipe-art", "vllm", "torch", "transformers", "peft", "torchao", "unsloth", "unsloth-zoo")}
    result["GPU"] = props.name
    result["CUDA"] = torch.version.cuda
    art_root = output / "art-workspace"
    name = "sft-load-pilot"
    internal = {
        "init_args": {"load_in_4bit": False, "load_in_8bit": False, "max_seq_length": 4096, "trust_remote_code": False},
        "peft_args": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "random_state": 20261011},
        "engine_args": {"dtype": "bfloat16", "max_model_len": 4096, "max_num_seqs": 1,
                        "gpu_memory_utilization": 0.45, "enforce_eager": True, "seed": 20261011,
                        "max_lora_rank": 8, "max_logprobs": 20, "enable_sleep_mode": True},
    }
    model = art.TrainableModel(name=name, project="research-sft-load-only", base_model=original["base_path"],
                               report_metrics=[], _internal_config=internal)
    model_dir = Path(dirs.get_model_dir(model, art_path=str(art_root)))
    staged = Path(dirs.get_step_checkpoint_dir(str(model_dir), 2))
    staged.mkdir(parents=True, exist_ok=False)
    for filename in original["checkpoint_files"]:
        shutil.copyfile(Path(original["checkpoint_path"]) / filename, staged / filename)
    # Copy only the verified flat adapter inventory; never move/modify Drive original.
    if artifact_inventory(staged) != original["checkpoint_files"]:
        raise ValueError("ART work copy differs from Drive source")
    tokenizer = transformers.AutoTokenizer.from_pretrained(original["base_path"], local_files_only=True, trust_remote_code=False)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != original["chat_template_sha256"]:
        raise ValueError("effective tokenizer differs from identity")
    install_monitor_workaround(backend_class)
    async with managed_backend(backend_class(in_process=True, path=str(art_root))) as backend:
        await asyncio.wait_for(model.register(backend, _openai_client_config={
            "server_args": {"host": "127.0.0.1", "tool_call_parser": "hermes", "enable_auto_tool_choice": True}
        }), timeout=600)
        if await model.get_step() != 2 or model.get_inference_name() != name + "@2":
            raise ValueError("ART did not select the explicit SFT checkpoint 2")
        service = backend._services[name]  # inspected ART 0.5.17 implementation, not a portable API promise
        state = service._state
        loaded = state.peft_model
        # Unsloth 2026.3.3 sets is_loaded_in_8bit=True even for dense models
        # (models/llama.py: DDP compatibility marker). Audit actual storage, not that flag.
        base_model = loaded.get_base_model()
        audit = {"is_loaded_in_4bit": bool(getattr(loaded, "is_loaded_in_4bit", False)),
                 "is_loaded_in_8bit_marker": bool(getattr(loaded, "is_loaded_in_8bit", False)),
                 "quantization_config_present": getattr(base_model.config, "quantization_config", None) is not None,
                 "parameter_dtypes": {}, "projection_classes": {}, "projection_count": 0}
        result["loader_quantization_audit"] = audit
        if audit["is_loaded_in_4bit"] or audit["quantization_config_present"]:
            raise ValueError("actual loader quantization flag/config not allowed")
        if service.config["init_args"].get("load_in_4bit") or service.config["init_args"].get("load_in_8bit"):
            raise ValueError("ART effective init args requested quantization")
        if service.config["engine_args"].get("quantization") is not None:
            raise ValueError("ART vLLM engine args requested quantization")
        allowed = {torch.float16, torch.bfloat16, torch.float32}
        for parameter_name, parameter in loaded.named_parameters():
            if (type(parameter) is not torch.nn.Parameter or type(parameter.data) is not torch.Tensor
                    or not parameter.is_floating_point() or parameter.dtype not in allowed):
                raise ValueError("non-dense/quantized parameter storage: " + parameter_name)
            dtype = str(parameter.dtype)
            audit["parameter_dtypes"][dtype] = audit["parameter_dtypes"].get(dtype, 0) + 1
        targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        for module_name, module in base_model.named_modules():
            if module_name.rsplit(".", 1)[-1] not in targets:
                continue
            dense = module.get_base_layer() if hasattr(module, "get_base_layer") else module
            class_name = type(dense).__module__ + "." + type(dense).__name__
            audit["projection_classes"][class_name] = audit["projection_classes"].get(class_name, 0) + 1
            if type(dense) is not torch.nn.Linear or dense.weight.dtype != torch.bfloat16:
                raise ValueError("expected dense BF16 base projection: " + module_name + " (" + class_name + ")")
            audit["projection_count"] += 1
        if audit["projection_count"] != base_model.config.num_hidden_layers * len(targets):
            raise ValueError("base projection audit coverage mismatch")
        if not audit["parameter_dtypes"]:
            raise ValueError("empty loader parameter audit")
        write_json(output / "loader_quantization_audit.json", audit)
        active = list(loaded.active_adapters)
        if len(active) != 1:
            raise ValueError("expected exactly one active SFT adapter in ART loader")
        result["ART_active_adapters"] = active
        runtime = peft.get_peft_model_state_dict(loaded, adapter_name=active[0])
        saved = safetensors.load_file(str(staged / "adapter_model.safetensors"), device="cpu")
        if set(runtime) != set(saved):
            raise ValueError("loaded LoRA tensor names differ")
        dtype_casts = {}
        for key, value in runtime.items():
            actual = value.detach().cpu()
            expected = saved[key].to(actual.dtype)
            if not torch.equal(actual, expected):
                raise ValueError("loaded LoRA differs beyond explicit dtype cast: " + key)
            if actual.dtype != saved[key].dtype:
                dtype_casts[key] = {"source": str(saved[key].dtype), "loaded": str(actual.dtype)}
        result["loaded_LoRA_tensor_count"] = len(runtime)
        result["loaded_LoRA_dtype_casts"] = dtype_casts
        result["ART_effective_config"] = service.config
        endpoint = model.inference_base_url
        if not endpoint or urlsplit(endpoint).hostname != "127.0.0.1":
            raise ValueError("ART inference endpoint must bind loopback only")
        async with openai.AsyncOpenAI(base_url=endpoint, api_key=model.inference_api_key, max_retries=0,
                                     http_client=httpx.AsyncClient(trust_env=False, timeout=120)) as client:
            models = (await client.models.list()).model_dump(mode="json")
            write_json(output / "served_models.json", models)
            card = next((x for x in models["data"] if x["id"] == name + "@2"), None)
            if card is None or card.get("root") != str(staged) or card.get("parent") != original["base_path"]:
                raise ValueError("vLLM served adapter root/parent identity mismatch")
            result["served_adapter_name"] = card["id"]
            result["served_adapter_path"] = card["root"]
            # Fixed teacher-forced TRAIN prefix: distinguish active adapter from Base.
            train, _ = load_registered_development(ROOT, split="train")
            probe_row = next(r for r in train if r["reference_depth"] == 1)
            tokenized = tokenize_expert(build_expert(probe_row), tokenizer, max_length=4096)
            eos_index = tokenized["assistant_token_ranges"][-1][1] - 1
            probe_ids = tokenized["input_ids"][:eos_index]
            probes = []
            for served in (original["base_path"], card["id"]):
                reply = await client.completions.create(model=served, prompt=probe_ids, temperature=0,
                                                        max_tokens=1, logprobs=20, seed=20261011,
                                                        extra_body={"ignore_eos": True, "skip_special_tokens": False})
                probes.append({"model": served, "prompt_token_ids": probe_ids, "target_token_id": tokenizer.eos_token_id,
                               "prefix": "first-train-depth1-before-terminal-assistant-EOS",
                               "decode": {"temperature": 0, "max_tokens": 1, "logprobs": 20, "seed": 20261011,
                                          "ignore_eos": True, "skip_special_tokens": False},
                               "response": reply.model_dump(mode="json")})
            write_json(output / "adapter_base_probe.json", probes)
            scores = [p["response"]["choices"][0]["logprobs"]["top_logprobs"][0] for p in probes]
            common = set(scores[0]) & set(scores[1])
            delta = max((abs(scores[0][k] - scores[1][k]) for k in common), default=0)
            result["adapter_base_common_top20_logprob_delta"] = delta
            if delta <= 1e-4:
                raise ValueError("fixed probe did not distinguish adapter from Base; do not claim active-weight verification")

            class DiagnosticBackend:
                receipt = {"backend": "art-vllm-load-only", "training_updates": 0}

                async def complete(self, request):
                    chat = copy.deepcopy(request["messages"])
                    for message in chat:
                        for call in message.get("tool_calls", []):
                            if isinstance(call["function"]["arguments"], str):
                                try:
                                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                                except (ValueError, TypeError) as exc:
                                    raise ModelProtocolError("invalid prior tool-call arguments JSON") from exc
                    prompt_ids = tokenizer.apply_chat_template(chat, tools=request["tools"], tokenize=True, add_generation_prompt=True)
                    if len(prompt_ids) > request["max_prompt_tokens"]:
                        raise PromptBudgetExceeded("prompt exceeds diagnostic limit")
                    api_chat = copy.deepcopy(request["messages"])
                    for message in api_chat:
                        for call in message.get("tool_calls", []):
                            if isinstance(call["function"]["arguments"], dict):
                                call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
                    reply = await client.chat.completions.create(
                        model=card["id"], messages=api_chat, tools=request["tools"],
                        temperature=request["temperature"], top_p=request["top_p"], max_tokens=request["max_new_tokens"],
                        seed=request["seed"], timeout=request["timeout_seconds"])
                    raw = reply.model_dump(mode="json")
                    if raw["usage"]["prompt_tokens"] != len(prompt_ids):
                        raise RuntimeError("server/local template prompt token count differs")
                    raw["local_prompt_token_ids"] = prompt_ids
                    return raw

                async def close(self):
                    pass  # enclosing client/backend contexts own shutdown

            adapter = DiagnosticBackend()
            # Scaffolded observation round trip, explicitly NOT model achievement.
            record = records[2]
            step = protocol_plan(record)[0][0]
            observation = ResearchEnvironment(record["environment_snapshot"]).execute(step["name"], step["arguments"])
            chat = model_messages(record) + [
                {"role": "assistant", "content": None, "tool_calls": [{"id": "scaffold-0", "type": "function", "function": step}]},
                {"role": "tool", "tool_call_id": "scaffold-0", "content": observation},
            ]
            request = {"messages": chat, "tools": TOOL_SCHEMAS, "temperature": 0.0, "top_p": 1.0,
                       "max_new_tokens": 128, "max_prompt_tokens": 4096, "seed": 20261011, "timeout_seconds": 60}
            reply = await adapter.complete(request)
            normalized, _, _ = normalize_response(reply, 1)
            write_json(output / "controlled_observation_probe.json", {"scripted_prefix": True, "counts_toward_model_metrics": False,
                       "request": request, "response": reply, "normalized": normalized})
            for row, slot in zip(records, plan_attempts(records, protocol), strict=True):
                if time.monotonic() - begun >= WALL_CAP:
                    raise TimeoutError("ART pilot wall deadline reached")
                outcome = await evaluate_one(row, slot, protocol, adapter)
                received.append(outcome)
                with (output / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            result["greedy_generated_tool_calls"] = sum(
                len(message.get("tool_calls", [])) for attempt in received
                for message in attempt["messages"] if message["role"] == "assistant")
            result["scaffolded_generated_tool_calls"] = len(normalized.get("tool_calls", []))
            result["generated_tool_parser_coverage"] = (result["greedy_generated_tool_calls"]
                                                       + result["scaffolded_generated_tool_calls"]) > 0
            if await model.get_step() != 2 or artifact_inventory(staged) != original["checkpoint_files"]:
                raise ValueError("ART step/adapter artifacts changed without authorization")
        result["ART_step_before_after"] = [2, 2]
    # Original remains immutable and is the authoritative diagnostic artifact.
    if time.monotonic() - begun > WALL_CAP:
        raise TimeoutError("ART pilot exceeded wall cap")
    if any(x["infra_failure"] for x in received):
        raise RuntimeError("diagnostic inference contains infra failures; retain all attempts")
    result["ART_vLLM_load_verified"] = True
    result["status"] = "load-and-transport-passed-quality-not-claimed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-art", action="store_true")
    parser.add_argument("--confirm-load-only", action="store_true")
    args = parser.parse_args()
    if args.run_art and not args.confirm_load_only:
        raise ValueError("ART loading requires explicit --confirm-load-only; no training authorized")
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("use a new output directory outside repository")
    if args.run_art and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("ART pilot requires clean source checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    result = {"status": "preparing", "training_updates": 0, "formal_training": False,
              "ART_vLLM_load_verified": False, "GPU_requested": args.run_art, "actual_allocation_time": None,
              "total_GPU_budget_verified": False, "started_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "pilot_source_sha256": file_hash(Path(__file__)), "wall_seconds_cap": WALL_CAP, "pid": os.getpid()}
    records, plan, received = [], [], []
    spec = identity = None
    try:
        write_json(args.output / "result.json", result)
        spec, identity, records, source, protocol = preflight(args.base_dir, args.adapter_dir)
        plan = plan_attempts(records, protocol)
        write_json(args.output / "plan.json", plan)
        write_json(args.output / "preflight.json", {"model_spec": spec, "model_identity": identity, "source": source,
                   "selected_sample_ids": [r["sample_id"] for r in records], "depths": [1, 2, 3],
                   "protocol": protocol, "protocol_sha256": digest(protocol), "diagnostic_only": True})
        if args.run_art:
            asyncio.run(run_art(args.output, identity, records, protocol, result, begun, received))
        else:
            result["status"] = "preflight-passed-no-model-load"
        verify_unchanged(spec, identity)
        if file_hash(Path(__file__)) != result["pilot_source_sha256"]:
            raise ValueError("pilot source changed during run")
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if plan:
            write_json(args.output / "ledger.json", complete_ledger(records, plan, received))
        result["planned_attempts"] = len(plan)
        result["returned_attempts"] = len(received)
        result["infra_failures"] = sum(x["infra_failure"] for x in received)
        result["missing_attempts"] = len(plan) - len(received)
        result["answer_correct"] = sum(x["checker"]["answer"]["correct"] for x in received)
        result["wall_seconds"] = time.monotonic() - begun
        result["pilot_process_wall_hours"] = result["wall_seconds"] / 3600
        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

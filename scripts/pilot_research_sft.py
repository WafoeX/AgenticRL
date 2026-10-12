#!/usr/bin/env python3
"""Explicit two-update A100 SFT pilot; default is tokenizer/identity-only preflight."""

import argparse
import gc
import importlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_model_identity import (
    artifact_inventory,
    file_hash,
    resolve_identity,
    verify_unchanged,
)
from src.research_schema import load_json
from src.research_sft_pilot import (
    CONFIG,
    prepare,
    tensor_fingerprint,
    train_updates,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    # Run directory is exclusively created. Status can advance only inside this new run.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_base(identity, torch, transformers):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        identity["base_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map=None, use_safetensors=True, attn_implementation="eager")
    if str(Path(model.config._name_or_path).resolve()) != identity["base_path"] or type(model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("actual loaded base identity mismatch")
    file_config = load_json(Path(identity["base_path"]) / "config.json")
    actual = model.config.to_dict()
    for key, value in file_config.items():
        if key in {"_name_or_path", "_commit_hash", "transformers_version", "torch_dtype"} or actual.get(key) == value:
            continue
        dormant = (key == "sliding_window" and type(value) is int and value == 32768
                   and type(file_config.get("use_sliding_window")) is bool and not file_config["use_sliding_window"]
                   and type(actual.get("use_sliding_window")) is bool and not actual["use_sliding_window"]
                   and "sliding_window" in actual and actual[key] is None)
        if not dormant:
            raise ValueError("runtime base config mismatch: " + key)
    model.to("cuda:0")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--confirm-pilot-only", action="store_true")
    parser.add_argument("--remaining-gpu-hours", type=float, help="Deprecated; ignored (no manual budget gate)")
    parser.add_argument("--gpu-allocated-at", help="Deprecated; ignored (no manual allocation-time gate)")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("run output must be outside repository/frozen evidence")
    budget = None
    if args.run_gpu:
        if not args.confirm_pilot_only:
            raise ValueError("GPU requires explicit pilot-only confirmation")
        budget = {"policy": "user-amendment-v2-no-manual-budget-gate", "total_GPU_budget_verified": False,
                  "actual_allocation_time": None, "prior_GPU_consumption": None,
                  "metering_scope": "this-pilot-process-only", "wall_seconds_cap": CONFIG["max_wall_seconds"],
                  "legacy_budget_arguments_ignored": args.remaining_gpu_hours is not None or args.gpu_allocated_at is not None}
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise ValueError("GPU pilot requires a clean versioned checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    result = {"status": "preparing", "planned_updates": 2 if args.run_gpu else 0, "actual_updates": 0,
              "GPU_requested": args.run_gpu, "training_performed": False, "budget": budget,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "pid": os.getpid(), "formal_training": False, "ART_vLLM_verified": False,
              "source_hashes": {name: file_hash(ROOT / name) for name in (
                  "scripts/pilot_research_sft.py", "src/research_sft_pilot.py", "src/research_sft.py",
                  "configs/research/sft_pilot_v1.json", "requirements-sft-pilot.txt")}}
    torch = None
    model = reloaded = base = fresh = None
    try:
        write_json(args.output / "result.json", result)
        receipt, experts, rows, batch = prepare(ROOT, args.base_dir)
        write_json(args.output / "preflight.json", receipt)
        write_json(args.output / "selected_experts.json", {"experts": experts})
        if not args.run_gpu:
            result["status"] = "preflight-passed-no-model-load"
            return
        # User removed the manual allocation gate; prior/session-wide costs remain unknown.
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("exactly one visible CUDA GPU required")
        props = torch.cuda.get_device_properties(0)
        if "A100" not in props.name or props.total_memory < 39 * 1024**3:
            raise ValueError("pilot requires A100 with at least 39 GiB visible memory")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 unavailable")
        torch.manual_seed(CONFIG["seed"])
        torch.cuda.manual_seed_all(CONFIG["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        result["runtime"] = {"GPU": props.name, "total_memory": props.total_memory, "dtype": "bfloat16",
                             "CUDA": torch.version.cuda, "python": sys.version,
                             "versions": {name: metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")}}
        identity = receipt["identity"]
        base = load_base(identity, torch, transformers)
        result["runtime"]["real_base_loaded"] = True
        result["runtime"]["model_class"] = type(base).__name__
        model = peft.get_peft_model(base, peft.LoraConfig(
            task_type="CAUSAL_LM", r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=0.0, target_modules=CONFIG["target_modules"], bias="none"))
        inputs = {key: torch.tensor(value, dtype=torch.long, device="cuda:0") for key, value in batch.items()}
        # Teacher-forced last-token logits on a fixed train prefix; not a model accuracy metric.
        probe = {key: value[:1, :rows[0]["token_count"] - 1] for key, value in inputs.items() if key != "labels"}

        def logits(current):
            current.eval()
            with torch.no_grad():
                return current(**probe, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()

        before_logits = logits(model)
        logs = args.output / "updates.jsonl"

        def emit(row):
            with logs.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            result["actual_updates"] = row["step"]
            result["training_performed"] = True
            write_json(args.output / "result.json", result)
            print(json.dumps(row, sort_keys=True), flush=True)

        result.update(train_updates(model, inputs, CONFIG, torch, deadline=begun + CONFIG["max_wall_seconds"], emit=emit))
        after_logits = logits(model)
        result["probe_logit_delta"] = (after_logits - before_logits).abs().max().item()
        if result["probe_logit_delta"] <= 0 or result["initial_adapter_sha256"] == result["final_adapter_sha256"]:
            raise ValueError("no observable adapter update")
        adapter = args.output / "checkpoint-2"
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        write_json(adapter / "research_checkpoint.json", {"step": 2, "base_fingerprint": identity["base_fingerprint"],
                                                          "base_revision": CONFIG["base_revision"]})
        spec = {"kind": "checkpoint", "base_path": identity["base_path"], "base_revision": CONFIG["base_revision"],
                "checkpoint_path": str(adapter), "checkpoint_step": 2}
        checkpoint_identity = resolve_identity(spec)
        state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        # Release the first model before loading the fresh base/adapter, not same-object reuse.
        model = base = None
        gc.collect()
        torch.cuda.empty_cache()
        fresh = load_base(identity, torch, transformers)
        reloaded = peft.PeftModel.from_pretrained(fresh, adapter, is_trainable=False)
        actual_state = {n: p.detach().cpu() for n, p in reloaded.named_parameters() if "lora_" in n}
        if set(actual_state) != set(state) or any(not torch.equal(state[n], actual_state[n]) for n in state):
            raise ValueError("saved/reloaded adapter tensors differ")
        reload_logits = logits(reloaded)
        result["reload_logit_max_difference"] = (after_logits - reload_logits).abs().max().item()
        if not torch.allclose(after_logits, reload_logits, atol=CONFIG["reload_logit_atol"], rtol=CONFIG["reload_logit_rtol"]):
            raise ValueError("fresh checkpoint reload changed behavior beyond declared bf16 tolerance")
        result["reloaded_adapter_sha256"] = tensor_fingerprint(actual_state, torch)
        if result["reloaded_adapter_sha256"] != result["final_adapter_sha256"]:
            raise ValueError("adapter tensor identity differs after reload")
        verify_unchanged(spec, checkpoint_identity)
        verify_unchanged({"kind": "base", "base_path": identity["base_path"], "base_revision": CONFIG["base_revision"],
                          "checkpoint_path": None, "checkpoint_step": None}, identity)
        write_json(args.output / "checkpoint_identity.json", checkpoint_identity)
        result["checkpoint_files"] = artifact_inventory(adapter)
        if time.monotonic() - begun > CONFIG["max_wall_seconds"]:
            raise TimeoutError("pilot exceeded total wall deadline; do not claim bounded success")
        if any(file_hash(ROOT / name) != sha for name, sha in result["source_hashes"].items()):
            raise ValueError("pilot source changed during run")
        result["status"] = "SFT-and-Transformers-reload-passed-ART-vLLM-pending"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        model = reloaded = base = fresh = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
        result["wall_seconds"] = time.monotonic() - begun
        result["pilot_process_wall_hours"] = result["wall_seconds"] / 3600
        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

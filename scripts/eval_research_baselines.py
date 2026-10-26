#!/usr/bin/env python3
"""Full validation only: Base or SFT, same existing CUDA greedy decoder.

Default is identity/plan preflight, not inference. No training, ART or test.
"""
import argparse
import asyncio
import gc
import importlib
import json
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_research_sft import load_base
from scripts.train_research_sft import append_json, spec_for
from src.research_eval_metrics import summarize_evaluation
from src.research_eval_protocol import plan_attempts, protocol_identity
from src.research_evaluation import complete_ledger, evaluate_one
from src.research_model_identity import file_hash, resolve_identity, verify_unchanged
from src.research_registration import load_registered_development
from src.research_schema import load_json
from src.research_sft_run import SFTValidationBackend, write_json

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
VERSIONS = {"torch": "2.8.0", "transformers": "4.57.1", "peft": "0.17.1", "safetensors": "0.8.0"}


def flow_metrics(rows):
    """Counts from actual executed tool events; failed early stops stay visible."""
    def group(selected):
        n = len(selected)
        calls = sum(len(r.get("tool_events", [])) for r in selected)
        turns = sum(len(r.get("requests", [])) for r in selected)
        return {"tasks": n, "tool_calls": calls, "generation_requests": turns,
                "mean_tool_calls": calls / n if n else None, "mean_turns": turns / n if n else None}
    success = [r for r in rows if r.get("checker", {}).get("answer", {}).get("correct", False)]
    return {"all_tasks": group(rows), "answer_correct_tasks": group(success),
            "note": "Successful-task means are conditional, not matched-task efficiency gains."}


class BaselineValidationBackend(SFTValidationBackend):
    """Reuse SFTValidationBackend._generate unchanged for BOTH arms."""
    def __init__(self, model, tokenizer, torch, identity, versions):
        parameters = list(model.parameters())
        if not parameters or any(str(p.device) != "cuda:0" for p in parameters):
            raise ValueError("explicit cuda:0 model required")
        if str(model.get_input_embeddings().weight.dtype) != "torch.bfloat16":
            raise ValueError("same bfloat16 base precision as SFT validation required")
        if identity["kind"] == "checkpoint":
            adapters = list(model.active_adapters)
            if adapters != ["default"]:
                raise ValueError("only the selected default SFT adapter is allowed")
        else:
            adapters = []
            if getattr(model, "peft_config", None) or any("lora_" in n for n, _ in model.named_parameters()):
                raise ValueError("Base must not contain a project adapter")
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self._template, self._lock, self._closed = tokenizer.chat_template, asyncio.Lock(), False
        self.receipt = {"backend": "transformers-cuda-baseline-comparison-v2", "real_model_loaded": True,
                        "identity": identity, "device": "cuda:0", "base_dtype": "bfloat16",
                        "versions": versions, "active_adapters": adapters, "source_weights_verified": True,
                        "decoder": "unchanged-SFTValidationBackend._generate"}


def load_backend(identity, protocol):
    versions = {p: metadata.version(p) for p in VERSIONS}
    if any(versions[p].split("+")[0] != v for p, v in VERSIONS.items()):
        raise ValueError("restore the existing SFT package versions, not the ART environment")
    torch, transformers = importlib.import_module("torch"), importlib.import_module("transformers")
    if not torch.cuda.is_available():
        raise ValueError("--run-gpu requires a CUDA device")
    tokenizer = transformers.AutoTokenizer.from_pretrained(identity["base_path"], local_files_only=True,
                                                           trust_remote_code=False, use_fast=True)
    model = load_base(identity, torch, transformers)
    if identity["kind"] == "checkpoint":
        peft = importlib.import_module("peft")
        model = peft.PeftModel.from_pretrained(model, identity["checkpoint_path"], is_trainable=False,
                                              local_files_only=True, use_safetensors=True)
        saved = importlib.import_module("safetensors.torch").load_file(
            str(Path(identity["checkpoint_path"]) / "adapter_model.safetensors"), device="cpu")
        actual = peft.get_peft_model_state_dict(model)
        if set(actual) != set(saved) or any(not torch.equal(actual[n].detach().cpu(), saved[n]) for n in saved):
            raise ValueError("loaded SFT adapter differs from saved weights")
    model.eval()
    return BaselineValidationBackend(model, tokenizer, torch, identity, versions)


async def run(args, *, loader=load_backend):
    if (args.arm == "B1") != (args.checkpoint_dir is not None):
        raise ValueError("B1 requires --checkpoint-dir; B0 must not receive an adapter")
    spec = spec_for(Path(args.base_dir), REVISION,
                    Path(args.checkpoint_dir) if args.checkpoint_dir else None, 10 if args.arm == "B1" else None)
    identity = resolve_identity(spec)
    records, dataset = load_registered_development(ROOT, split="validation")
    if len(records) != 180:
        raise ValueError("full validation must contain 180 scenarios")
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    protocol["modes"] = ["deterministic"]
    plan = plan_attempts(records, protocol)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), ROOT / "src/research_sft_run.py", ROOT / "src/research_evaluation.py",
                    ROOT / "src/research_eval_backend.py", ROOT / "src/research_checker.py", ROOT / "src/research_answers.py"]
    manifest = {"version": "baseline-validation-v2", "arm": args.arm, "state": "preflight-only",
                "model_spec": spec, "identity": identity, "dataset": dataset, "protocol": protocol_identity(protocol),
                "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
                "loader_receipt": None, "training_performed": False, "test_used": False, "automatic_retries": 0}
    write_json(output / "plan.json", plan)
    write_json(output / "manifest.json", manifest)
    if not args.run_gpu:
        return manifest
    backend, received = None, []
    started = time.monotonic()
    (output / "attempts.jsonl").touch(exist_ok=False)
    try:
        manifest["state"] = "loading"
        write_json(output / "manifest.json", manifest)
        backend = loader(identity, protocol)
        manifest.update(state="running", loader_receipt=backend.receipt)
        write_json(output / "manifest.json", manifest)
        for slot in plan:
            row = await evaluate_one(records[slot["scenario_index"]], slot, protocol, backend)
            append_json(output / "attempts.jsonl", row)
            received.append(row)
            if len(received) % 10 == 0:
                print(f"{args.arm}: {len(received)}/180", flush=True)
        verify_unchanged(spec, identity)
        manifest["state"] = "completed"
    except BaseException as exc:
        manifest.update(state="failed", error_type=type(exc).__name__)
        raise
    finally:
        try:
            if backend is not None:
                await backend.close()
        except BaseException as exc:
            manifest.update(state="failed", close_error_type=type(exc).__name__)
            raise
        finally:
            backend = None
            gc.collect()
            ledger = complete_ledger(records, plan, received)
            summary = summarize_evaluation(ledger, protocol)
            if manifest["state"] != "completed":
                summary["comparison_valid"] = False
            summary["tool_flow"] = flow_metrics(ledger)
            write_json(output / "summary.json", summary)
            (output / "ledger.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in ledger))
            manifest["wall_seconds"] = time.monotonic() - started
            manifest["artifacts"] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
            write_json(output / "manifest.json", manifest)
    if not summary["comparison_valid"]:
        raise ValueError("infrastructure/missing output: preserved run is not a valid comparison")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["B0", "B1"], required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-gpu", action="store_true", help="Explicitly execute 180 validation tasks; no training/test")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({k: result[k] for k in ("arm", "state", "training_performed", "test_used")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

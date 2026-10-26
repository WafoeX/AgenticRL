#!/usr/bin/env python3
"""Explicit short SFT + full validation selection; default tokenizer-only preflight.

Never runs GRPO or test. Continuation requires a new output/config and explicit
--resume-from pointing to a committed last checkpoint, not a selected best alias.
"""

import argparse
import asyncio
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_research_sft import load_base  # noqa: E402
from src.research_eval_metrics import summarize_evaluation  # noqa: E402
from src.research_eval_protocol import plan_attempts, protocol_identity  # noqa: E402
from src.research_evaluation import complete_ledger, evaluate_one  # noqa: E402
from src.research_model_identity import file_hash, resolve_identity, verify_unchanged  # noqa: E402
from src.research_registration import load_registered_development  # noqa: E402
from src.research_schema import digest, load_json  # noqa: E402
from src.research_sft import build_corpus  # noqa: E402
from src.research_sft_pilot import tensor_fingerprint  # noqa: E402
from src.research_sft_run import (  # noqa: E402
    CONFIG_PATH,
    SFTValidationBackend,
    batch_indices,
    lora_parameters,
    optimizer_update,
    restore_resume,
    resume_contract,
    save_resume,
    select_checkpoint,
    stage_grpo_starts,
    validate_config,
    write_json,
)


def append_json(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def spec_for(base: Path, revision: str, checkpoint: Path | None = None, step: int | None = None) -> dict:
    return {"kind": "base" if checkpoint is None else "checkpoint", "base_path": str(base.resolve()),
            "base_revision": revision, "checkpoint_path": str(checkpoint.resolve()) if checkpoint else None,
            "checkpoint_step": step}


def load_resume_source(path: Path) -> dict:
    """Complete checkpoint, not arbitrary .art/latest/best/diagnostic step2."""
    path = path.resolve(strict=True)
    marker = load_json(path.parent / f"continuation-{load_json(path / 'research_checkpoint.json')['step']}.json")
    if marker["version"] != "research-sft-continuation-v1" or marker["checkpoint_path"] != str(path):
        raise ValueError("resume checkpoint marker/path mismatch; use original committed last")
    # This checkpoint was the last successfully saved update, even if later eval failed.
    for name, sha in marker["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or file_hash(path / name) != sha:
            raise ValueError("resume checkpoint artifacts changed")
    expected = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "continuation.json"}
    if expected != set(marker["files"]):
        raise ValueError("resume checkpoint missing/extra state files")
    parent = load_json(path.parent / "result.json")
    if parent.get("last_committed_step") != marker["step"] or parent.get("last_checkpoint") != str(path):
        raise ValueError("resume only from parent's last committed checkpoint, not an earlier best")
    if resolve_identity(marker["model_spec"]) != marker["identity"]:
        raise ValueError("resume weights/base changed")
    return marker


def prepare(base: Path, config: dict) -> tuple:
    records, source = load_registered_development(ROOT, split="train")
    validation, val_source = load_registered_development(ROOT, split="validation")
    if len(records) != 1200 or len(validation) != 180:
        raise ValueError("registered corpus cardinality changed")
    spec = spec_for(base, config["base_revision"])
    identity = resolve_identity(spec)
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        identity["base_path"], local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.get_chat_template() != tokenizer.chat_template:
        raise ValueError("ambiguous tokenizer template")
    import hashlib
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != identity["chat_template_sha256"]:
        raise ValueError("runtime chat template differs")
    _, rows, summary = build_corpus(records, tokenizer, max_length=config["max_length"])
    if summary["accepted"] != 1200 or summary["rejected_count"] != 0:
        raise ValueError("no silent dropping/truncating of train examples: " + json.dumps(summary))
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    protocol["modes"] = ["deterministic"]
    receipt = {"source": source, "validation_source": val_source, "identity": identity,
               "tokenized_rows_sha256": digest(rows), "corpus": summary,
               "protocol": protocol_identity(protocol), "train_count": 1200, "validation_count": 180}
    verify_unchanged(spec, identity)
    return receipt, rows, validation, tokenizer, protocol


async def validate_candidate(output: Path, spec: dict, identity: dict, train_model, tokenizer, records: list,
                             protocol: dict, versions: dict, torch, config: dict) -> dict:
    output.mkdir(exist_ok=False)
    plan = plan_attempts(records, protocol)
    write_json(output / "plan.json", plan)
    received = []
    backend = fresh = reloaded = None
    status = {"state": "loading", "identity": identity, "protocol": protocol_identity(protocol),
              "loader_receipt": None, "fresh_reload_tensor_verified": False}
    write_json(output / "manifest.json", status)
    (output / "attempts.jsonl").touch(exist_ok=False)
    failure = None
    try:
        fresh = load_base(identity, torch, importlib.import_module("transformers"))
        peft = importlib.import_module("peft")
        reloaded = peft.PeftModel.from_pretrained(fresh, identity["checkpoint_path"], is_trainable=False,
                                                 local_files_only=True, use_safetensors=True)
        expected = {n: p.detach().cpu() for n, p in lora_parameters(train_model).items()}
        actual = {n: p.detach().cpu() for n, p in reloaded.named_parameters() if "lora_" in n}
        if set(expected) != set(actual) or any(not torch.equal(expected[n], actual[n]) for n in expected):
            raise ValueError("fresh reload adapter tensors differ")
        # Compare teacher-forced logits on a train-only prefix; not a capability score.
        train_model.eval()
        reloaded.eval()
        prefix = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id, tokenizer.eos_token_id]], device="cuda:0")
        with torch.no_grad():
            before = train_model(input_ids=prefix, use_cache=False).logits.float()
            after = reloaded(input_ids=prefix, use_cache=False).logits.float()
        delta = (before - after).abs().max().item()
        if not torch.allclose(before, after, atol=0.02, rtol=0.002):
            raise ValueError("fresh reload logits differ beyond pilot bf16 tolerance")
        status.update(fresh_reload_tensor_verified=True, reload_logit_max_difference=delta,
                      tensor_fingerprint=tensor_fingerprint(actual, torch))
        del expected, actual, before, after
        backend = SFTValidationBackend(reloaded, tokenizer, torch, identity, versions)
        status.update(state="running", loader_receipt=backend.receipt)
        write_json(output / "manifest.json", status)

        def save(row):
            append_json(output / "attempts.jsonl", row)
            received.append(row)
            if len(received) % 10 == 0:
                print(f"validation step={spec['checkpoint_step']} returned={len(received)}/180", flush=True)

        # The CPU cohort runner deliberately accepts only CPU receipts. Reuse its
        # per-attempt tool/oracle path, not a fabricated CPU receipt for this GPU backend.
        for slot in plan:
            save(await evaluate_one(records[slot["scenario_index"]], slot, protocol, backend))
        verify_unchanged(spec, identity)
        status["state"] = "completed"
    except BaseException as exc:
        failure = exc
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            await backend.close()
        backend = reloaded = fresh = None
        gc.collect()
        torch.cuda.empty_cache()
        ledger = complete_ledger(records, plan, received)
        summary = summarize_evaluation(ledger, protocol)
        if failure is not None:
            summary["comparison_valid"] = False
        for row in ledger:
            append_json(output / "ledger.jsonl", row)
        write_json(output / "summary.json", summary)
        status["artifacts"] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
        write_json(output / "manifest.json", status)
    if failure is not None:
        raise failure
    overall = summary["evaluations"]["deterministic"]["overall"]
    row = {"step": spec["checkpoint_step"], "scenarios": overall["planned_scenarios"],
           "comparison_valid": summary["comparison_valid"],
           "answer_correct": overall["deterministic_answer_accuracy"]["numerator"],
           "dynamic_valid": overall["dynamic_reference_dependency_all"]["numerator"],
           "identity": identity, "model_spec": spec, "validation_manifest_sha256": file_hash(output / "manifest.json")}
    if not row["comparison_valid"] or row["scenarios"] != config["validation_scenarios"]:
        raise ValueError("validation infra/missing invalidates selection; no automatic retry")
    return row


def save_checkpoint(output: Path, base: Path, step: int, model, optimizer, torch, config: dict,
                    contract: dict, result: dict) -> tuple:
    checkpoint = output / f"checkpoint-{step}"
    checkpoint.mkdir(exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
    write_json(checkpoint / "research_checkpoint.json", {"step": step,
               "base_fingerprint": contract["data_and_tokenizer"]["identity"]["base_fingerprint"],
               "base_revision": config["base_revision"]})
    spec = spec_for(base, config["base_revision"], checkpoint, step)
    identity = resolve_identity(spec)
    save_resume(checkpoint / "continuation-state", model, optimizer, torch, step=step, contract=contract, cuda=True)
    marker = {"version": "research-sft-continuation-v1", "checkpoint_path": str(checkpoint.resolve()),
              "step": step, "config": config, "contract": contract, "model_spec": spec, "identity": identity,
              "files": {p.relative_to(checkpoint).as_posix(): file_hash(p) for p in checkpoint.rglob("*") if p.is_file()}}
    # Keep the commit marker outside the flat adapter inventory (no circular hash).
    write_json(output / f"continuation-{step}.json", marker)
    result.update(last_committed_step=step, last_checkpoint=str(checkpoint.resolve()))
    write_json(output / "result.json", result)
    return spec, identity


async def execute(args, output: Path, config: dict, parent: dict | None, result: dict) -> None:
    receipt, rows, validation, tokenizer, protocol = prepare(args.base_dir, config)
    write_json(output / "preflight.json", receipt)
    start = parent["step"] if parent else 0
    write_json(output / "training_plan.json", {"config": config, "start_step": start,
        "steps": [{"step": s, "sample_ids": [rows[i]["sample_id"] for i in batch_indices(1200, s, config)]}
                  for s in range(start + 1, config["terminal_step"] + 1)],
        "first_round_full_epoch": False, "test_used": False})
    if not args.run_gpu:
        result["state"] = "preflight-passed-no-model-load"
        return
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name or props.total_memory < 39 * 1024**3 or not torch.cuda.is_bf16_supported():
        raise ValueError("A100 40GB with bf16 required")
    versions = {n: metadata.version(n) for n in ("torch", "transformers", "peft", "safetensors")}
    contract = resume_contract(config, receipt, versions)
    contract["source_hashes"] = result["source_hashes"]
    if parent and contract != parent["contract"]:
        raise ValueError("resume requires identical data/base/template/runtime versions")
    result.update(GPU=props.name, versions=versions, resume_contract=contract, state="training")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    model = optimizer = base = None
    try:
        base = load_base(receipt["identity"], torch, importlib.import_module("transformers"))
        peft = importlib.import_module("peft")
        if parent:
            model = peft.PeftModel.from_pretrained(base, parent["checkpoint_path"], is_trainable=True,
                                                  local_files_only=True, use_safetensors=True)
        else:
            model = peft.get_peft_model(base, peft.LoraConfig(task_type="CAUSAL_LM", r=config["lora_r"],
                lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                target_modules=config["target_modules"], bias="none"))
        parameters = lora_parameters(model)
        optimizer = torch.optim.AdamW(parameters.values(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        if parent:
            restore_resume(Path(parent["checkpoint_path"]) / "continuation-state", model, optimizer, torch,
                           expected_contract=contract, device="cuda:0")
        result["initial_adapter_tensor_sha256"] = tensor_fingerprint(parameters, torch)
        candidates = []
        for step in range(start + 1, config["terminal_step"] + 1):
            group = [rows[i] for i in batch_indices(1200, step, config)]
            update = optimizer_update(model, optimizer, group, config, torch, pad_id=tokenizer.pad_token_id, device="cuda:0")
            update.update(step=step, sample_ids=[r["sample_id"] for r in group],
                          elapsed_seconds=time.monotonic() - result["monotonic_start"])
            append_json(output / "updates.jsonl", update)
            result.update(actual_updates_this_run=step - start, actual_step=step, training_performed=True)
            print(json.dumps(update, sort_keys=True), flush=True)
            if step in config["candidate_steps"]:
                spec, identity = save_checkpoint(output, args.base_dir, step, model, optimizer, torch, config, contract, result)
                # Fresh validation model construction must not perturb continuation RNG.
                with torch.random.fork_rng(devices=[0]):
                    candidate = await validate_candidate(output / f"validation-{step}", spec, identity, model, tokenizer,
                                                         validation, protocol, versions, torch, config)
                candidates.append(candidate)
                write_json(output / "candidates.json", candidates)
            write_json(output / "result.json", result)
        best = select_checkpoint(candidates, config["candidate_steps"])
        selection = {"config": config, "best": best, "last": candidates[-1], "candidates": candidates,
                     "selection_scope": "current-versioned-run-candidates-only", "test_used": False}
        write_json(output / "selection.json", selection)
        # Copy into an exclusive, run-local pair of independent workspaces. No .art/optimizer inherited.
        starts = stage_grpo_starts(best["model_spec"], output / "grpo-starts", selection=best)
        result.update(state="short-SFT-selected-shared-starts-prepared", selected_step=best["step"],
                      shared_start_fingerprint=starts["source"]["fingerprint"],
                      actual_GRPO_load_verified=False, ART_vLLM_this_checkpoint_verified=False)
        verify_unchanged(spec_for(args.base_dir, config["base_revision"]), receipt["identity"])
    finally:
        parameters = {}
        model = optimizer = base = None
        gc.collect()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_PATH)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--confirm-short-sft", action="store_true")
    args = parser.parse_args()
    if args.run_gpu and not args.confirm_short_sft:
        raise ValueError("GPU requires explicit --confirm-short-sft")
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(args.base_dir.resolve()) or output.exists():
        raise ValueError("new external output directory required; no overwrite/resume-in-place")
    config = load_json(args.config)
    parent = load_resume_source(args.resume_from) if args.resume_from else None
    if parent and output.is_relative_to(Path(parent["checkpoint_path"]).parent):
        raise ValueError("continuation must use a separate run directory outside the source run")
    validate_config(config, parent=parent["config"] if parent else None, start_step=parent["step"] if parent else 0)
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    source_files = [CONFIG_PATH, "src/research_sft_run.py", "scripts/train_research_sft.py",
                    "src/research_sft.py", "src/research_sft_pilot.py", "scripts/pilot_research_sft.py",
                    "src/research_model_identity.py", "src/research_evaluation.py", "src/research_eval_protocol.py",
                    "src/research_eval_backend.py", "src/research_eval_metrics.py", "src/research_answers.py",
                    "src/research_checker.py", "src/research_environment.py", "src/research_schema.py",
                    "src/research_registration.py", "src/research_generator.py", "src/research_audit.py",
                    "configs/research/evaluation_v1.json", "configs/research/preregistration_v1.json",
                    "configs/research/preregistration_v1.lock.json", "requirements-sft-pilot.txt"]
    result = {"version": "research-sft-run-v1", "state": "preparing", "config": config,
              "config_sha256": digest(config), "started_at": datetime.now(timezone.utc).isoformat(),
              "monotonic_start": begun, "actual_updates_this_run": 0, "actual_step": parent["step"] if parent else 0,
              "last_committed_step": None, "last_checkpoint": None, "GPU_requested": args.run_gpu,
              "training_performed": False, "test_used": False, "GRPO_training_started": False,
              "parent_continuation": parent, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_hashes": {p: file_hash(ROOT / p) for p in source_files},
              "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
              "total_GPU_budget_verified": False, "metering_scope": "process-only-not-allocation-idle-or-prior-runs",
              "automatic_retries": 0}
    try:
        write_json(output / "result.json", result)
        asyncio.run(execute(args, output, config, parent, result))
        if any(file_hash(ROOT / p) != sha for p, sha in result["source_hashes"].items()):
            raise ValueError("source changed during SFT run")
    except BaseException as exc:
        result.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(ended_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - begun)
        result["process_wall_hours"] = result["wall_seconds"] / 3600
        write_json(output / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

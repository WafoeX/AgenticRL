"""Separate final-test cohort. Default: byte-only preflight; never train."""
import argparse
import asyncio
import gc
import json
import os
import shutil
import statistics
import time
from pathlib import Path

from scripts.eval_research_baselines import REVISION, flow_metrics, load_backend
from scripts.train_research_sft import spec_for
from src.research_answers import reject_constant, unique_object
from src.research_eval_metrics import (
    cluster_interval,
    paired_difference,
    summarize_evaluation,
)
from src.research_evaluation import RecordedBackend, complete_ledger, evaluate_one
from src.research_model_identity import file_hash, resolve_identity, verify_unchanged
from src.research_registration import verify_registration
from src.research_schema import digest, load_json, validate_record
from src.research_sft_run import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/research/final_test_v1.json"
LOCK = "configs/research/final_test_v1.lock.json"
CONFIRM = "RUN_FROZEN_FOUR_MODEL_TEST_960"


def preflight(root=ROOT):
    """Hash/count test bytes only. Do not deserialize any test record here."""
    registration = verify_registration(root)
    lock = load_json(root / LOCK)
    for name, sha in lock["files"].items():
        if file_hash(root / name) != sha:
            raise ValueError("final-test frozen source/config changed: " + name)
    cfg = load_json(root / CONFIG)
    if (cfg["version"] != "four-model-final-test-v1" or cfg["gpu_execution_authorized"] is not False
            or cfg["scenarios"] != 240 or cfg["planned_tasks"] != 960
            or len(cfg["models"]) != 4 or cfg["automatic_retries"] != 0
            or registration["test_sha256"] != cfg["test_sha256"]):
        raise ValueError("invalid final cohort annex")
    protocol = load_json(root / cfg["protocol_source"])
    protocol["modes"] = ["deterministic"]
    # evaluate_one does not use the old development planner. No old gate is changed.
    return cfg, protocol, {"state": "preflight-only", "test_labels_parsed": False,
                          "model_inference": False, "GPU_started": False,
                          "config_sha256": file_hash(root / CONFIG), "lock_sha256": file_hash(root / LOCK),
                          "test_sha256": cfg["test_sha256"], "planned_tasks": 960}


def load_final_records(cfg, root=ROOT):
    """Called only after explicit execution approval and all model identities checked."""
    path = root / "data/research_v1_frozen/test.jsonl"
    if file_hash(path) != cfg["test_sha256"]:
        raise ValueError("test bytes changed")
    try:
        records = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                   for line in path.read_text().splitlines()]
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError("invalid frozen test JSONL") from exc
    if len(records) != 240:
        raise ValueError("full frozen test required")
    for record in records:
        validate_record(record)
        if record["split"] != "test":
            raise ValueError("mixed test split")
    if any(len({r[key] for r in records}) != 240 for key in ("sample_id", "semantic_group_id")):
        raise ValueError("duplicate test scenario/group")
    return records


def final_plan(records, protocol):
    """Explicit final-only schedule; never disguise test as validation."""
    if not records or any(r["split"] != "test" for r in records):
        raise ValueError("final planner requires test")
    slots = []
    for i, r in enumerate(records):
        slot = {k: r[k] for k in ("sample_id", "semantic_group_id", "family", "reference_depth",
                                  "dynamic_dependency", "generalization", "split")}
        slot.update(scenario_index=i, mode="deterministic", index=0)
        slot["attempt_id"] = "final-attempt-" + digest([r["sample_id"], digest(protocol), "four-model-final-test-v1"])
        slots.append(slot)
    return slots


def save(path, value, output, backup):
    """Atomic local write + fsync, Drive copy, then real reopen/hash both copies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    target = backup / path.relative_to(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(path, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(target)
    if file_hash(path) != file_hash(target) or load_json(path) != load_json(target):
        raise ValueError("local/Drive readback mismatch")


def read_prefix(directory, plan, backup):
    paths = sorted((directory / "attempts").glob("*.json"))
    if [p.name for p in paths] != [f"{i:04d}.json" for i in range(len(paths))] or len(paths) > len(plan):
        raise ValueError("noncontiguous/extra saved tasks")
    rows = []
    for i, path in enumerate(paths):
        if file_hash(path) != file_hash(backup / "attempts" / path.name):
            raise ValueError("local/Drive saved tasks differ")
        row = load_json(path)
        if any(row.get(k) != v for k, v in plan[i].items()) or row["infra_failure"] or not row["received"]:
            raise ValueError("saved task identity/infra invalid; do not retry")
        rows.append(row)
    return rows


async def verify_rows(records, plan, protocol, rows):
    """Re-execute raw responses/checker, not a model; reject edited/duplicate scores."""
    replay = RecordedBackend({"replay_only": True}, rows)
    for i, row in enumerate(rows):
        actual = await evaluate_one(records[i], plan[i], protocol, replay)
        if actual != {k: v for k, v in row.items() if k != "resources"}:
            raise ValueError("saved outcome does not replay exactly")


def comparison(arms):
    baseline = arms[0]
    paired = [paired_difference(baseline, rows) for rows in arms[1:]]
    for result, rows in zip(paired, arms[1:], strict=True):
        result["improved_ids"] = [r["sample_id"] for b, r in zip(baseline, rows, strict=True)
                                  if r["checker"]["answer"]["correct"] and not b["checker"]["answer"]["correct"]]
        result["regressed_ids"] = [r["sample_id"] for b, r in zip(baseline, rows, strict=True)
                                   if b["checker"]["answer"]["correct"] and not r["checker"]["answer"]["correct"]]
    rates = [sum(r["checker"]["answer"]["correct"] for r in rows) / len(rows) for rows in arms[1:]]
    delta = [sum(int(rows[i]["checker"]["answer"]["correct"]) for rows in arms[1:]) / 3
             - int(baseline[i]["checker"]["answer"]["correct"]) for i in range(len(baseline))]
    return {"primary": "mean-three-GRPO-minus-SFT30", "scenarios": len(delta),
            "GRPO_mean": statistics.mean(rates), "GRPO_sample_sd": statistics.stdev(rates),
            "mean_difference": statistics.mean(delta), "paired_scenario_ci95": cluster_interval(delta),
            "per_seed_paired": paired, "note": "CI conditional on these fixed models; scenario bootstrap, not 720 independent tasks. Three seeds do not estimate SFT training variance."}


def load_final_backend(identity, protocol):
    backend = load_backend(identity, protocol)
    torch = backend._torch
    if "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("frozen final test requires the same A100 environment")
    if any(p.requires_grad for p in backend._model.parameters()):
        raise ValueError("final test must have no trainable parameters")
    backend.receipt["purpose"] = "frozen-four-model-final-test-no-optimizer"
    backend.receipt["GPU_name"] = torch.cuda.get_device_name(0)
    return backend


async def run(args, *, loader=load_final_backend):
    cfg, protocol, receipt = preflight()
    if not args.run_gpu:
        return receipt
    if (args.confirm != CONFIRM or not args.balance_note or not args.wall_seconds or args.wall_seconds <= 420):
        raise ValueError("explicit approval, actual-balance note and budget greater than 420s required")
    started = time.monotonic()
    output, backup = Path(args.output).resolve(), Path(args.backup).resolve()
    if output == backup or output.is_relative_to(backup) or backup.is_relative_to(output):
        raise ValueError("separate nonnested local and Drive directories required")
    identities = []
    for model in cfg["models"]:
        spec = spec_for(Path(args.base_dir), REVISION, Path(args.drive_root) / model["checkpoint"], model["step"])
        identity = resolve_identity(spec)
        if identity["fingerprint"] != model["fingerprint"] or identity["base_fingerprint"] != cfg["base_fingerprint"]:
            raise ValueError("frozen checkpoint/base differs: " + model["arm"])
        identities.append((spec, identity))
    binding = {"preflight": receipt, "models": [i[1] for i in identities], "protocol": protocol,
               "training_performed": False, "automatic_retries": 0}
    if args.resume:
        manifest = load_json(output / "manifest.json")
        if manifest["binding"] != binding or manifest["state"] not in {"paused-clean-boundary", "completed"}:
            raise ValueError("only a verified clean-boundary pause may resume; failed/unknown runs need investigation")
        if file_hash(output / "manifest.json") != file_hash(backup / "manifest.json"):
            raise ValueError("manifest backup differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        backup.mkdir(parents=True, exist_ok=False)
        manifest = {"binding": binding, "state": "prepared", "sessions": [], "test_consumed": True}
    manifest["sessions"].append({"balance_note": args.balance_note, "wall_seconds_cap": args.wall_seconds})
    save(output / "manifest.json", manifest, output, backup)
    records = load_final_records(cfg)
    plan = final_plan(records, protocol)
    save(output / "plan.json", {"slots": plan}, output, backup)
    completed_arms = []
    try:
        for model, (spec, identity) in zip(cfg["models"], identities, strict=True):
            directory = output / model["arm"]
            rows = read_prefix(directory, plan, backup / model["arm"])
            await verify_rows(records, plan, protocol, rows)
            backend = None
            try:
                if len(rows) < len(plan):
                    if time.monotonic() - started >= args.wall_seconds - 420:
                        manifest["state"] = "paused-clean-boundary"
                        break
                    manifest.update(state="loading", active_arm=model["arm"])
                    save(output / "manifest.json", manifest, output, backup)
                    backend = loader(identity, protocol)
                    save(directory / "loader.json", backend.receipt, output, backup)
                for slot in plan[len(rows):]:
                    if time.monotonic() - started >= args.wall_seconds - 420:
                        manifest["state"] = "paused-clean-boundary"
                        break
                    manifest.update(state="running", active_attempt=slot["attempt_id"])
                    save(output / "manifest.json", manifest, output, backup)
                    if backend is None:
                        raise RuntimeError("missing loaded evaluation backend")
                    torch = backend._torch
                    torch.cuda.reset_peak_memory_stats(0)
                    before = time.monotonic()
                    row = await evaluate_one(records[slot["scenario_index"]], slot, protocol, backend)
                    row["resources"] = {"wall_seconds": time.monotonic() - before,
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                    save(directory / "attempts" / f"{len(rows):04d}.json", row, output, backup)
                    rows.append(row)
                    if row["infra_failure"]:
                        raise RuntimeError("infrastructure failure retained; stop entire cohort, no retry")
                    if len(rows) % 10 == 0:
                        print(f"{model['arm']}: {len(rows)}/240", flush=True)
                if backend is not None:
                    verify_unchanged(spec, identity)
            finally:
                if backend is not None:
                    torch = backend._torch
                    await backend.close()
                    backend = None
                    gc.collect()
                    torch.cuda.empty_cache()
            # Reopen real saved files, not the in-memory count.
            rows = read_prefix(directory, plan, backup / model["arm"])
            await verify_rows(records, plan, protocol, rows)
            ledger = complete_ledger(records, plan, rows)
            summary = summarize_evaluation(ledger, protocol)
            if len(rows) != 240:
                summary["comparison_valid"] = False
                summary["comparison_invalid_reason"] = "incomplete-final-arm; only clean-boundary same-cohort resume allowed, no automatic rerun"
            summary["tool_flow"] = flow_metrics(ledger)
            summary["task_wall_seconds"] = sum(r.get("resources", {}).get("wall_seconds", 0) for r in rows)
            save(directory / "summary.json", summary, output, backup)
            if len(rows) != 240:
                manifest["state"] = "paused-clean-boundary"
                break
            completed_arms.append(rows)
        if len(completed_arms) == 4:
            save(output / "comparison.json", comparison(completed_arms), output, backup)
            manifest["state"] = "completed"
    except BaseException as exc:
        manifest.update(state="failed-stop-investigate", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest["sessions"][-1]["measured_wall_seconds"] = time.monotonic() - started
        save(output / "manifest.json", manifest, output, backup)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--balance-note")
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--base-dir")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--output")
    parser.add_argument("--backup")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.run_gpu and result["state"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

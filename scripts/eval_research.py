#!/usr/bin/env python3
"""Isolated development research eval. Default preflight; real CPU load requires --run-cpu."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_research_dataset import legacy_hashes  # noqa: E402
from src.research_answers import ANSWER_VERSION, reject_constant, unique_object  # noqa: E402
from src.research_checker import TRACE_VERSION  # noqa: E402
from src.research_environment import canonical_json  # noqa: E402
from src.research_eval_backend import EvaluationBackend, TransformersCPUBackend  # noqa: E402
from src.research_eval_metrics import METRICS_VERSION, summarize_evaluation  # noqa: E402
from src.research_eval_protocol import (  # noqa: E402
    load_development_records,
    plan_attempts,
    protocol_identity,
    validate_protocol,
)
from src.research_evaluation import (  # noqa: E402
    RecordedBackend,
    complete_ledger,
    evaluate_one,
    evaluate_records,
)
from src.research_model_identity import (  # noqa: E402
    file_hash,
    resolve_identity,
    verify_receipt,
    verify_unchanged,
)
from src.research_schema import load_json  # noqa: E402

SOURCE_PATHS = [
    "src/research_answers.py", "src/research_checker.py", "src/research_environment.py", "src/research_schema.py",
    "src/research_model_identity.py", "src/research_eval_protocol.py", "src/research_eval_backend.py",
    "src/research_evaluation.py", "src/research_eval_metrics.py", "scripts/eval_research.py",
]


def source_identity() -> dict:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError("cannot identify source base commit")
    return {"base_commit": result.stdout.strip(), "working_tree_sha256": {p: file_hash(ROOT / p) for p in SOURCE_PATHS},
            "note": "Hashes identify actual dirty/untracked source; base_commit alone does not contain it."}


def checked_output(value: str) -> Path:
    output = (ROOT / value).resolve()
    allowed = (ROOT / "results/research").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("output must be a dedicated results/research subdirectory")
    return output


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError("cannot read raw evaluation JSONL") from exc


def report_markdown(manifest: dict, summary: dict) -> str:
    real = (manifest.get("loader_receipt") or {}).get("real_model_loaded", False)
    title = "本地 CPU 模型评测" if real else "CPU 模拟响应合同验收（不是模型准确率）"
    lines = [f"# {title}", "", f"run ID: `{manifest['run_id']}`；状态：`{manifest['state']}`。",
             "无 GPU、训练、reward 评分、自动重试或 test。", "",
             "实际加载身份/文件 SHA-256、协议/hash、全部计划 attempts 与原始消息保存在 manifest/plan/attempts/ledger。", "",
             f"可比较：`{summary['comparison_valid']}`；infra/缺失时保留全部分母，必须明确新 run ID 重跑整次。", "",
             "| mode | 场景 | 计划 attempts | 有效 | infra | 缺失返回 | 最终答案指标（分子/分母） |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for mode, evaluation in summary["evaluations"].items():
        stats = evaluation["overall"]
        metric = stats["deterministic_answer_accuracy" if mode == "deterministic" else "observed_answer_pass_at_8"]
        lines.append(f"| {mode} | {stats['planned_scenarios']} | {stats['planned_attempts']} | {stats['valid_attempts']} | {stats['infra_failures']} | {stats['missing_returned_attempts']} | {metric['numerator']}/{metric['denominator']} |")
    lines += ["", "主答案不受严格参考路线/reward门控；strict/semantic/dependency/bypass与任务分层另列于summary。",
              "Wilson区间按独立语义场景；sample rollout区间按场景聚类bootstrap，不把8次采样当8个任务。",
              "真实CPU后端的max_time为软停止、结束当前forward；超时取消会等待生成线程退出，不遗留后台推理。",
              "本轮模拟合同验收不等于真实权重/ART-vLLM加载成功，温度0不保证跨环境字节确定性。", ""]
    return "\n".join(lines)


async def run_local_evaluation(spec: dict, protocol: dict, records: list[dict], dataset: dict, output: Path,
                               *, run_cpu: bool = False,
                               loader: Callable[[dict, dict, dict], EvaluationBackend] = TransformersCPUBackend) -> dict:
    validate_protocol(protocol)
    if output.exists():
        raise ValueError("output exists; refusing overwrite/resume/append")
    identity = resolve_identity(spec)
    plan = plan_attempts(records, protocol)
    before = legacy_hashes()
    manifest = {"version": "research-evaluation-manifest-v1", "run_id": output.name, "state": "preflight-only",
                "model_spec": spec, "model_identity": identity, "loader_receipt": None, "dataset": dataset,
                "protocol": protocol_identity(protocol), "checker_versions": {"answer": ANSWER_VERSION, "trace": TRACE_VERSION, "metrics": METRICS_VERSION},
                "source": source_identity(), "legacy_hashes_before": before, "legacy_hashes_after": before,
                "auto_retry_count": 0, "GPU_started": False, "test_used": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "manifest.json", manifest)
    if not run_cpu:
        return manifest
    received = []
    backend = None
    failure = None
    (output / "attempts.jsonl").touch(exist_ok=False)

    def save_result(result):
        with (output / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        received.append(result)

    try:
        manifest["state"] = "loading"
        write_json(output / "manifest.json", manifest)
        backend = loader(spec, identity, protocol)
        verify_receipt(identity, backend.receipt)
        manifest["loader_receipt"] = backend.receipt
        manifest["state"] = "running"
        write_json(output / "manifest.json", manifest)
        await evaluate_records(records, protocol, identity, backend, on_result=save_result)
        verify_unchanged(spec, identity)
        manifest["state"] = "completed"
    except BaseException as exc:
        failure = exc
        manifest["state"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        manifest["fatal_error_type"] = type(exc).__name__
    finally:
        if backend is not None:
            try:
                await backend.close()
            except BaseException as exc:
                manifest["close_error_type"] = type(exc).__name__
                if failure is None:
                    failure = exc
                    manifest["state"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
                    manifest["fatal_error_type"] = type(exc).__name__
        manifest["legacy_hashes_after"] = legacy_hashes()
        ledger = complete_ledger(records, plan, received)
        summary = summarize_evaluation(ledger, protocol)
        summary["execution_state"] = manifest["state"]
        if failure is not None or manifest["legacy_hashes_after"] != before:
            summary["comparison_valid"] = False
            summary["comparison_invalid_reason"] = "interrupted/failed/identity-or-legacy-mutation; inspect preserved original run"
        (output / "ledger.jsonl").write_text("".join(canonical_json(row) + "\n" for row in ledger), encoding="utf-8")
        write_json(output / "summary.json", summary)
        (output / "REPORT.md").write_text(report_markdown(manifest, summary), encoding="utf-8")
        manifest["artifacts"] = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                                 for p in sorted(output.iterdir()) if p.name != "manifest.json" and p.is_file()}
        write_json(output / "manifest.json", manifest)
    if failure is not None:
        raise failure
    if manifest["legacy_hashes_after"] != before:
        raise ValueError("legacy data/tools changed")
    return manifest


async def audit_saved_run(output: Path) -> dict:
    manifest = load_json(output / "manifest.json")
    if manifest["state"] != "completed" or manifest["loader_receipt"] is None:
        raise ValueError("audit requires completed loader-attested run; partial/preflight logs are retained but not successful")
    if manifest["source"] != source_identity():
        raise ValueError("current source differs from recorded evaluator/checker")
    if set(manifest["artifacts"]) != {"plan.json", "attempts.jsonl", "ledger.jsonl", "summary.json", "REPORT.md"}:
        raise ValueError("missing/unknown artifact checksum entries")
    if manifest["checker_versions"] != {"answer": ANSWER_VERSION, "trace": TRACE_VERSION, "metrics": METRICS_VERSION}:
        raise ValueError("checker version identity mismatch")
    for name, info in manifest["artifacts"].items():
        if Path(name).name != name or file_hash(output / name) != info["sha256"] or (output / name).stat().st_size != info["bytes"]:
            raise ValueError("saved artifact bytes differ from manifest")
    protocol = manifest["protocol"]["configuration"]
    if protocol_identity(protocol) != manifest["protocol"]:
        raise ValueError("protocol/hash mismatch")
    directory = Path(manifest["dataset"]["path"]).parent
    rows, dataset = load_development_records(directory, split=manifest["dataset"]["split"])
    if dataset["sha256"] != manifest["dataset"]["sha256"] or dataset["manifest_sha256"] != manifest["dataset"]["manifest_sha256"]:
        raise ValueError("source dataset changed")
    by_id = {row["sample_id"]: row for row in rows}
    records = [by_id[key] for key in manifest["dataset"]["selected_sample_ids"]]
    if manifest["dataset"]["selected_count"] != len(records) or manifest["dataset"]["source_count"] != len(rows):
        raise ValueError("dataset selection denominator mismatch")
    identity = manifest["model_identity"]
    verify_unchanged(manifest["model_spec"], identity)
    verify_receipt(identity, manifest["loader_receipt"])
    plan = plan_attempts(records, protocol)
    try:
        saved_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (OSError, ValueError) as exc:
        raise ValueError("invalid saved attempt plan") from exc
    if saved_plan != plan:
        raise ValueError("attempt plan changed")
    raw = read_jsonl(output / "attempts.jsonl")
    complete_ledger(records, plan, raw)  # reject duplicates/unknown IDs/stratum substitutions
    replay = RecordedBackend(manifest["loader_receipt"], raw)
    recomputed = []
    try:
        for original in raw:
            result = await evaluate_one(records[original["scenario_index"]], plan[next(i for i, p in enumerate(plan) if p["attempt_id"] == original["attempt_id"])], protocol, replay)
            # Original infrastructure exception type is an opaque diagnostic string.
            for old, new in zip(original["requests"], result["requests"], strict=True):
                if isinstance(old["error"], str) and old["error"].startswith("backend_exception:") and isinstance(new["error"], str) and new["error"].startswith("backend_exception:"):
                    new["error"] = old["error"]
            if original != result:
                raise ValueError("raw-response replay differs from saved messages/provenance/checker results")
            recomputed.append(result)
    finally:
        await replay.close()
    ledger = complete_ledger(records, plan, recomputed)
    summary = summarize_evaluation(ledger, protocol)
    summary["execution_state"] = manifest["state"]
    if read_jsonl(output / "ledger.jsonl") != ledger or load_json(output / "summary.json") != summary:
        raise ValueError("ledger/denominators/checker/metric recomputation differs")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/evaluation_v1.json")
    parser.add_argument("--data", default="data/research_v1_candidate")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--limit", type=int, default=None, help="explicit development subset, not full benchmark")
    parser.add_argument("--output", default="results/research/research_eval_dev_v1")
    parser.add_argument("--model-kind", choices=("base", "checkpoint"), default="base")
    parser.add_argument("--base-dir")
    parser.add_argument("--base-revision")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--run-cpu", action="store_true", help="opt in to real local CPU inference; NEVER starts CUDA/backend service")
    parser.add_argument("--audit", action="store_true", help="read-only raw-response/provenance/checker/metric replay; no model load")
    args = parser.parse_args()
    try:
        output = checked_output(args.output)
        if args.audit:
            if args.run_cpu:
                raise ValueError("audit cannot also run inference")
            asyncio.run(audit_saved_run(output))
            print("PASS: saved raw responses, ledger/checkers and all denominators reproduce; no inference")
            return 0
        if not args.base_dir or not args.base_revision:
            raise ValueError("explicit local --base-dir and pinned --base-revision required")
        protocol = load_json(ROOT / args.config)
        records, dataset = load_development_records(ROOT / args.data, split=args.split, limit=args.limit)
        spec = {"kind": args.model_kind, "base_path": args.base_dir, "base_revision": args.base_revision,
                "checkpoint_path": args.checkpoint_dir, "checkpoint_step": args.checkpoint_step}
        manifest = asyncio.run(run_local_evaluation(spec, protocol, records, dataset, output, run_cpu=args.run_cpu))
        print("PASS: " + manifest["state"] + " at " + str(output.relative_to(ROOT)))
        print("NOTE: no GPU/test/training; preflight is not successful real-model loading")
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print("ERROR: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

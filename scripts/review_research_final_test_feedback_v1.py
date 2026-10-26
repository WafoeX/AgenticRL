"""Offline audit of the frozen four-model final-test feedback; no model forward/GPU.

Reads existing raw token IDs/responses and runs the unchanged executor/checker.
Only tokenizer assets are loaded, offline. This is NOT a new test evaluation.
"""
import argparse
import asyncio
import csv
import hashlib
import importlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from scripts.eval_research_baselines import VERSIONS, flow_metrics
from scripts.eval_research_final_test_v1 import (
    comparison,
    final_plan,
    load_final_records,
    preflight,
)
from src.research_answers import reject_constant, unique_object
from src.research_eval_metrics import summarize_evaluation
from src.research_evaluation import RecordedBackend, evaluate_one
from src.research_model_identity import artifact_inventory, file_hash
from src.research_schema import digest

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_SHA = "cd434cd4e036d613710996817145708eeea7093e16e18ebe9fb9b73c8bef1868"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(data):
    try:
        return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid/duplicate/nonfinite feedback JSON") from exc


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def correct(row):
    return int(row["checker"]["answer"]["correct"])


def bootstrap(values):
    # Independent implementation of the preregistered 1000-replicate endpoints.
    rng = random.Random(20261011)
    means = []
    for _ in range(1000):
        sample = [rng.randrange(len(values)) for _ in values]
        means.append(sum(values[i] for i in sample) / len(values))
    means.sort()
    return [means[24], means[974]]


def bucket(rows):
    n = len(rows)
    dynamic = [r for r in rows if r["dynamic_dependency"]]
    return {"n": n, "answer_correct": sum(map(correct, rows)),
            "strict": sum(r["checker"]["trace"]["strict_complete"] for r in rows),
            "semantic": sum(r["checker"]["trace"]["semantic_valid"] for r in rows),
            "dynamic_valid": sum(r["checker"]["trace"]["dynamic_status"] == "valid" for r in dynamic),
            "dynamic_n": len(dynamic), "calls": sum(len(r["tool_events"]) for r in rows),
            "requests": sum(len(r["requests"]) for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "task_wall_seconds": sum(r["resources"]["wall_seconds"] for r in rows),
            "multi_errors": dict(sorted(Counter(e for r in rows for e in r["errors"]).items()))}


def independent_pair(a, b):
    index = sorted(range(len(a)), key=lambda i: a[i]["sample_id"])
    delta = [correct(b[i]) - correct(a[i]) for i in index]
    return {"improved": delta.count(1), "regressed": delta.count(-1),
            "difference": sum(delta) / len(delta), "ci95": bootstrap(delta)}


def paired_costs(arms, indices):
    return {"tasks": len(indices), "arms": [bucket([rows[i] for i in indices]) for rows in arms]}


class CaptureReplay(RecordedBackend):
    """Rebuild insertion order from the frozen executor, not sorted JSON objects."""
    def __init__(self, rows):
        super().__init__({"replay_only": True}, rows)
        self.live_requests = {}

    async def complete(self, request):
        self.live_requests[(request["attempt_id"], request["turn"])] = request
        return await super().complete(request)


async def review(feedback, output, base_dir):
    require(file_hash(feedback) == FEEDBACK_SHA, "unexpected original feedback bytes")
    cfg, protocol, prepared = preflight()
    with ZipFile(feedback) as source:
        base_files = json_bytes(source.read("run/manifest.json"))["binding"]["models"][0]["base_files"]
    require(artifact_inventory(base_dir) == base_files, "local tokenizer/base artifacts differ from frozen evaluation")
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        str(base_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    records = load_final_records(cfg)
    plan = final_plan(records, protocol)
    output.mkdir(parents=True, exist_ok=False)
    arms = [m["arm"] for m in cfg["models"]]
    all_rows, summaries, receipts, inventory = [], {}, {}, {}
    token_requests = 0
    serialized_prompt_length_deltas = Counter()
    max_prompt, max_completion, max_turns = 0, 0, 0
    with ZipFile(feedback) as z:
        names = z.namelist()
        expected = {"run/manifest.json", "run/plan.json", "run/comparison.json", "readback.json", "execution.log",
                    "nvidia_after.txt", "final_test_v1.json", "final_test_v1.lock.json", "bundle_manifest.json"}
        for arm in arms:
            expected |= {f"run/{arm}/loader.json", f"run/{arm}/summary.json"}
            expected |= {f"run/{arm}/attempts/{i:04d}.json" for i in range(240)}
        require(len(names) == len(set(names)) and set(names) == expected, "extra/missing/duplicate archive paths")
        require(z.testzip() is None, "ZIP CRC failure")
        for name in names:
            data = z.read(name)
            inventory[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name in ["final_test_v1.json", "final_test_v1.lock.json"]:
            require(z.read(name) == (ROOT / "configs/research" / name).read_bytes(), "annex/source lock differs")
        with ZipFile(ROOT / "results/research/final_test_preparation_v1/final_test_v1_bundle.zip") as original:
            require(z.read("bundle_manifest.json") == original.read("bundle_manifest.json"), "delivery manifest differs")
            bundle = json_bytes(z.read("bundle_manifest.json"))
            require(all(hashlib.sha256(original.read(n)).hexdigest() == sha for n, sha in bundle["files"].items()), "original package changed")
        manifest = json_bytes(z.read("run/manifest.json"))
        require(manifest["state"] == "completed" and manifest["test_consumed"] is True, "not a completed final test")
        binding = manifest["binding"]
        require(binding["protocol"] == protocol and binding["preflight"] == prepared, "effective protocol/preflight mismatch")
        require(binding["automatic_retries"] == 0 and binding["training_performed"] is False, "unexpected training/retry")
        require(json_bytes(z.read("run/plan.json")) == {"slots": plan}, "different task schedule")
        sessions = manifest["sessions"]
        require(len(sessions) == 1 and sessions[0]["wall_seconds_cap"] == 10800 and sessions[0]["measured_wall_seconds"] < 10800,
                "unexpected sessions/budget")
        readback = json_bytes(z.read("readback.json"))
        require(readback["matched"] and readback["complete"], "remote readback did not pass")
        for j, arm in enumerate(arms):
            require(readback["counts"][arm] == {"files": 240, "unique_attempts": 240, "infra": 0}, "remote incomplete count")
            identity = binding["models"][j]
            require(identity["fingerprint"] == cfg["models"][j]["fingerprint"], "wrong frozen checkpoint")
            require(identity["base_fingerprint"] == cfg["base_fingerprint"], "different Base")
            require(digest(identity["base_files"]) == identity["base_fingerprint"], "invalid Base fingerprint")
            require(digest(identity["checkpoint_files"]) == identity["checkpoint_fingerprint"], "invalid adapter fingerprint")
            require(digest({k: identity[k] for k in ("kind", "base_fingerprint", "checkpoint_fingerprint", "checkpoint_step")})
                    == identity["fingerprint"], "invalid overall fingerprint")
            require(identity["checkpoint_path"] == "/content/drive/MyDrive/" + cfg["models"][j]["checkpoint"], "different checkpoint path")
            loader = json_bytes(z.read(f"run/{arm}/loader.json"))
            require(loader["identity"] == identity and loader["real_model_loaded"] is True
                    and loader["source_weights_verified"] is True and loader["active_adapters"] == ["default"]
                    and loader["device"] == "cuda:0" and loader["base_dtype"] == "bfloat16"
                    and loader["GPU_name"] == "NVIDIA A100-SXM4-40GB" and loader["versions"] == VERSIONS
                    and loader["purpose"] == "frozen-four-model-final-test-no-optimizer"
                    and loader["decoder"] == "unchanged-SFTValidationBackend._generate", "incompatible loader receipt")
            receipts[arm] = {k: v for k, v in loader.items() if k != "identity"}
            rows = [json_bytes(z.read(f"run/{arm}/attempts/{i:04d}.json")) for i in range(240)]
            require(len({r["attempt_id"] for r in rows}) == 240, "duplicate attempt")
            replay = CaptureReplay(rows)
            for i, row in enumerate(rows):
                actual = await evaluate_one(records[i], plan[i], protocol, replay)
                require(actual == {k: v for k, v in row.items() if k != "resources"}, "raw outcome replay mismatch")
                require(all(row[k] == v for k, v in plan[i].items()), "task slot mismatch")
                require(row["received"] and row["valid_attempt"] and not row["infra_failure"], "infra/missing/invalid task")
                require(row["termination"] == "finished", "unexpected unfinished task")
                require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                            for v in row["resources"].values()), "invalid resource counters")
                max_turns = max(max_turns, len(row["requests"]))
                for event in row["requests"]:
                    request, response = event["request"], event["raw_response"]
                    if event["error"] is not None or response is None:
                        raise ValueError("request failure")
                    ids = response["raw_token_ids"]
                    require(ids and all(type(v) is int and 0 <= v < len(tokenizer) for v in ids), "invalid raw token IDs")
                    eos = ids[-1] == tokenizer.eos_token_id
                    text = tokenizer.decode(ids[:-1] if eos else ids, skip_special_tokens=False)
                    require(response["choices"][0]["raw_text"] == text, "raw text not decoded from stored token IDs")
                    require(response["choices"][0]["finish_reason"] == ("stop" if eos else "length"), "finish reason mismatch")
                    fresh_request = replay.live_requests[(request["attempt_id"], request["turn"])]
                    prompt = tokenizer.apply_chat_template(fresh_request["messages"], tools=fresh_request["tools"],
                                                          chat_template=tokenizer.chat_template, tokenize=True,
                                                          add_generation_prompt=True)
                    serialized_prompt = tokenizer.apply_chat_template(request["messages"], tools=request["tools"],
                                                                      chat_template=tokenizer.chat_template, tokenize=True,
                                                                      add_generation_prompt=True)
                    serialized_prompt_length_deltas[str(len(serialized_prompt) - len(prompt))] += 1
                    require(response["usage"] == {"prompt_tokens": len(prompt), "completion_tokens": len(ids)}
                            and event["usage"] == response["usage"], "usage does not match actual tokenizer IDs")
                    token_requests += 1
                    max_prompt, max_completion = max(max_prompt, len(prompt)), max(max_completion, len(ids))
            await replay.close()
            summary = summarize_evaluation(rows, protocol)
            summary["tool_flow"] = flow_metrics(rows)
            summary["task_wall_seconds"] = sum(r["resources"]["wall_seconds"] for r in rows)
            require(summary == json_bytes(z.read(f"run/{arm}/summary.json")), "saved summary not reproducible")
            counts = bucket(rows)
            overall = summary["evaluations"]["deterministic"]["overall"]
            require(counts["answer_correct"] == overall["deterministic_answer_accuracy"]["numerator"]
                    and counts["strict"] == overall["deterministic_strict_rate"]["numerator"]
                    and counts["dynamic_n"] == overall["dynamic_reference_dependency_all"]["denominator"], "independent counts mismatch")
            counts.update(answer_rate=counts["answer_correct"] / 240, mean_calls=counts["calls"] / 240,
                          mean_completion_tokens=counts["completion_tokens"] / 240,
                          peak_allocated_GiB=max(r["resources"]["peak_allocated_bytes"] for r in rows) / 2**30,
                          peak_reserved_GiB=max(r["resources"]["peak_reserved_bytes"] for r in rows) / 2**30)
            summaries[arm] = counts
            all_rows.append(rows)
            print(arm, "240 exact raw replays; token IDs/usage checked", flush=True)
        paired = comparison(all_rows)
        require(paired == json_bytes(z.read("run/comparison.json")), "saved pairing not reproducible")
        main_delta = [sum(correct(rows[i]) for rows in all_rows[1:]) / 3 - correct(all_rows[0][i]) for i in range(240)]
        require(bootstrap(main_delta) == paired["paired_scenario_ci95"]
                and statistics.mean(main_delta) == paired["mean_difference"], "independent primary estimate mismatch")
        for seed_rows, entry in zip(all_rows[1:], paired["per_seed_paired"], strict=True):
            independent = independent_pair(all_rows[0], seed_rows)
            require(independent == {"improved": entry["second_only_correct"], "regressed": entry["first_only_correct"],
                                    "difference": entry["second_minus_first"], "ci95": entry["ci95"]}, "independent paired delta mismatch")
        nvidia = z.read("nvidia_after.txt").decode()
        require("No running processes found" in nvidia and "0MiB /" in nvidia, "GPU ending snapshot not empty")
    strata = {}
    for field in ("family", "reference_depth", "generalization"):
        strata[field] = {str(key): {arm: bucket([r for r in rows if r[field] == key])
                                  for arm, rows in zip(arms, all_rows, strict=True)}
                         for key in sorted({r[field] for r in all_rows[0]})}
    common = [i for i in range(240) if all(correct(rows[i]) for rows in all_rows)]
    common_improved = [i for i in range(240) if not correct(all_rows[0][i]) and all(correct(rows[i]) for rows in all_rows[1:])]
    outside = [i for i in range(240) if all_rows[0][i]["family"] != "country_density_convert"]
    outside_delta = [sum(correct(rows[i]) for rows in all_rows[1:]) / 3 - correct(all_rows[0][i]) for i in outside]
    costs = {"all_four_common_correct": paired_costs(all_rows, common),
             "all_three_common_improvement": {"n": len(common_improved),
               "families": dict(Counter(all_rows[0][i]["family"] for i in common_improved)),
               "sample_ids": [all_rows[0][i]["sample_id"] for i in common_improved]},
             "excluding_country_density_convert_exploratory": {"n": len(outside),
                "counts": [sum(correct(rows[i]) for i in outside) for rows in all_rows],
                "mean_difference": statistics.mean(outside_delta), "paired_ci95": bootstrap(outside_delta)}}
    regressions = sorted({sample_id for p in paired["per_seed_paired"] for sample_id in p["regressed_ids"]})
    case_ids = [*regressions]
    for family in ("country_density_convert", "measurement_scale", "distance_offset_convert"):
        case_ids.append(next((all_rows[0][i]["sample_id"] for i in common_improved if all_rows[0][i]["family"] == family), ""))
    cases = []
    for sample_id in filter(None, case_ids):
        i = next(i for i, r in enumerate(all_rows[0]) if r["sample_id"] == sample_id)
        cases.append({"sample_id": sample_id, "task": records[i]["task"], "family": records[i]["family"],
                      "generalization": records[i]["generalization"], "gold": records[i]["oracle_answer"],
                      "arms": {arm: {"correct": bool(correct(rows[i])), "final_answer": rows[i]["final_answer"],
                                     "errors": rows[i]["errors"], "messages": rows[i]["messages"], "trace": rows[i]["trace"]}
                               for arm, rows in zip(arms, all_rows, strict=True)}})
    verification = {"status": "passed-offline-final-test-feedback-review", "feedback": str(feedback.relative_to(ROOT)),
                    "feedback_sha256": FEEDBACK_SHA, "archive_entries": len(inventory),
                    "source_lock_files_verified": len(json_bytes((ROOT / "configs/research/final_test_v1.lock.json").read_bytes())["files"]),
                    "raw_outcomes_exact_replayed": 960, "raw_token_requests_checked": token_requests,
                    "all_summaries_and_comparison_exact": True, "independent_counts_and_bootstrap": True,
                    "remote_readback_receipt": readback, "sessions": sessions,
                    "max_prompt_tokens": max_prompt, "max_completion_tokens_per_turn": max_completion,
                    "max_generation_turns": max_turns, "loader_receipts": receipts,
                    "stored_JSON_prompt_length_minus_source_rebuilt_histogram": dict(serialized_prompt_length_deltas),
                    "prompt_rebuild_note": "Saved JSON sorts object keys; retokenizing that JSON directly changes tool-schema key order. All usage checks use fresh requests rebuilt by the frozen executor (including original insertion order), with semantic equality checked by RecordedBackend. Initial local audit assumption failed; no remote protocol/score changes.",
                    "gpu_end_snapshot_empty": True, "new_model_forward": False, "GPU_started_locally": False,
                    "training_performed_locally": False, "dependencies_installed": False,
                    "evidence_limits": ["Drive equality is a remote readback receipt, not a second local Drive copy",
                      "No independent local loading of remote adapter tensors; loader evidence is receipt plus matched locked source",
                      "Session wall time is not full Colab allocation/installation/idle accounting",
                      "No per-arm loading/synchronization wall split; only per-task timing and overall session timing",
                      "Outcome raw-response replay verifies saved behavior, not re-running the stochastic hardware model"]}
    for filename, value in [("verification.json", verification), ("inventory.json", inventory),
                            ("metrics.json", summaries), ("paired.json", paired), ("strata.json", strata),
                            ("costs_and_robustness.json", costs), ("cases.json", cases)]:
        write(output / filename, value)
    with (output / "per_scenario.csv").open("x", newline="") as stream:
        fields = ["arm", "sample_id", "family", "depth", "generalization", "correct", "strict", "dynamic_status",
                  "tool_calls", "requests", "completion_tokens", "prompt_tokens", "wall_seconds", "errors"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, rows in zip(arms, all_rows, strict=True):
            for r in rows:
                writer.writerow({"arm": arm, "sample_id": r["sample_id"], "family": r["family"], "depth": r["reference_depth"],
                    "generalization": r["generalization"], "correct": correct(r), "strict": r["checker"]["trace"]["strict_complete"],
                    "dynamic_status": r["checker"]["trace"]["dynamic_status"], "tool_calls": len(r["tool_events"]),
                    "requests": len(r["requests"]), "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"],
                    "wall_seconds": r["resources"]["wall_seconds"], "errors": ";".join(r["errors"])})
    print(json.dumps({"state": verification["status"], "correct": [summaries[a]["answer_correct"] for a in arms],
                      "mean_gain_pp": paired["mean_difference"] * 100, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    options = parser.parse_args()
    # Import/script use is rooted in the repository; no sys.path or package mutation.
    require(sys.flags.optimize == 0, "run audit with ordinary Python")
    asyncio.run(review(options.feedback.resolve(), options.output.resolve(), options.base_dir.resolve()))

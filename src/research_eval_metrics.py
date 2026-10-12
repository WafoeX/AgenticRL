"""Raw counts/all denominators, task-level Wilson CI and paired/cluster statistics."""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

METRICS_VERSION = "research-eval-metrics-v1"
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_REPLICATES = 1000


def wilson(successes: int, total: int) -> list[float] | None:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None
    z = 1.959963984540054
    p, denominator = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def cluster_interval(values: Sequence[float]) -> list[float] | None:
    """Resample semantic scenarios, NEVER treat their 8 attempts as 8 tasks."""
    if not values:
        return None
    rng, n = random.Random(BOOTSTRAP_SEED), len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_REPLICATES))
    return [samples[24], samples[974]]


def rate(successes: int, total: int, *, task_ci: bool = False) -> dict:
    return {"numerator": successes, "denominator": total, "rate": successes / total if total else None,
            "ci95": wilson(successes, total) if task_ci else None}


def summarize_bucket(rows: list[dict], mode: str) -> dict:
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["sample_id"]].append(row)
    def answer(r):
        return r["checker"]["answer"]["correct"]

    def strict(r):
        return r["checker"]["trace"]["strict_complete"]

    def semantic(r):
        return r["checker"]["trace"]["semantic_valid"]

    valid = [r for r in rows if r["valid_attempt"]]
    dynamic = [r for r in rows if r["dynamic_dependency"]]
    answer_count = sum(answer(r) for r in rows)
    task_answers = sum(any(answer(r) for r in members) for members in tasks.values())
    task_strict = sum(any(strict(r) for r in members) for members in tasks.values())
    result = {
        "planned_scenarios": len(tasks), "planned_attempts": len(rows), "received_attempts": sum(r["received"] for r in rows),
        "valid_attempts": len(valid), "infra_failures": sum(r["infra_failure"] for r in rows),
        "missing_output_attempts": sum(r["termination"] == "missing_output" for r in rows),
        "missing_returned_attempts": sum(not r["received"] for r in rows),
        "answer_rollout_accuracy_all": rate(answer_count, len(rows)),
        "answer_rollout_accuracy_valid": rate(sum(answer(r) for r in valid), len(valid)),
        "strict_rollout_rate_all": rate(sum(strict(r) for r in rows), len(rows)),
        "semantic_rollout_rate_all": rate(sum(semantic(r) for r in rows), len(rows)),
        "dynamic_reference_dependency_all": rate(sum(r["checker"]["trace"]["dynamic_status"] == "valid" for r in dynamic), len(dynamic)),
        "dynamic_alternative_bypass_attempts": sum(r["checker"]["trace"]["dynamic_status"] == "alternative_bypass" for r in dynamic),
        "primary_errors": dict(sorted(Counter(r["primary_error"] for r in rows).items())),
        "multi_label_errors": dict(sorted(Counter(error for r in rows for error in r["errors"]).items())),
        "completion_tokens_known": sum(r["completion_tokens"] for r in rows),
        "prompt_tokens_known": sum(r["prompt_tokens"] for r in rows),
    }
    if mode == "deterministic":
        result["deterministic_answer_accuracy"] = rate(task_answers, len(tasks), task_ci=True)
        result["deterministic_strict_rate"] = rate(task_strict, len(tasks), task_ci=True)
    else:
        result["observed_answer_pass_at_8"] = rate(task_answers, len(tasks), task_ci=True)
        result["observed_strict_pass_at_8"] = rate(task_strict, len(tasks), task_ci=True)
        means = [sum(answer(r) for r in members) / len(members) for members in tasks.values()]
        result["answer_rollout_accuracy_all"]["ci95"] = cluster_interval(means)
        result["answer_rollout_accuracy_all"]["ci_method"] = "semantic-scenario-cluster-percentile-bootstrap"
        result["complete_8_returned_scenarios"] = sum(len(members) == 8 and all(r["received"] for r in members) for members in tasks.values())
        result["complete_8_valid_scenarios"] = sum(len(members) == 8 and all(r["valid_attempt"] for r in members) for members in tasks.values())
    return result


def summarize_evaluation(ledger: list[dict], protocol: dict) -> dict:
    if len({r["attempt_id"] for r in ledger}) != len(ledger):
        raise ValueError("duplicate attempt in metric input")
    scenario_sets = [{r["sample_id"] for r in ledger if r["mode"] == mode} for mode in protocol["modes"]]
    if not ledger or any(not keys or keys != scenario_sets[0] for keys in scenario_sets) or any(r["mode"] not in protocol["modes"] for r in ledger):
        raise ValueError("incomplete planned denominator: absent mode/scenarios")
    modes = {}
    for mode in protocol["modes"]:
        rows = [r for r in ledger if r["mode"] == mode]
        groups = defaultdict(list)
        for row in rows:
            groups[row["sample_id"]].append(row)
        expected = 1 if mode == "deterministic" else 8
        if any(len(members) != expected or {r["index"] for r in members} != set(range(expected)) for members in groups.values()):
            raise ValueError("incomplete planned denominator: use complete_ledger first")
        strata = {}
        for field in ("family", "reference_depth", "dynamic_dependency", "generalization"):
            buckets = defaultdict(list)
            for row in rows:
                buckets[str(row[field])].append(row)
            strata[field] = {key: summarize_bucket(members, mode) for key, members in sorted(buckets.items())}
        modes[mode] = {"overall": summarize_bucket(rows, mode), "strata": strata}
    invalid = any(r["infra_failure"] or not r["received"] or r["termination"] == "missing_output" for r in ledger)
    return {"version": METRICS_VERSION, "evaluations": modes, "comparison_valid": not invalid,
            "comparison_invalid_reason": "infra-or-missing-output: retain original and explicitly rerun whole run under new ID" if invalid else None,
            "attempt_policy": protocol["infra_policy"], "task_ci_method": "Wilson95-semantic-scenarios",
            "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "unit": "semantic-scenario"},
            "interpretation": "All-attempt/scenario rates remain conservative when infra/missing affects an invalid comparison."}


def paired_difference(first: list[dict], second: list[dict], *, mode: str = "deterministic") -> dict:
    def outcomes(rows):
        groups = defaultdict(list)
        for row in rows:
            if row["mode"] == mode:
                groups[row["sample_id"]].append(row)
        expected = 1 if mode == "deterministic" else 8
        if any(len(members) != expected for members in groups.values()):
            raise ValueError("incomplete paired attempts")
        return {key: any(r["checker"]["answer"]["correct"] for r in members) for key, members in groups.items()}

    if mode not in {"deterministic", "sample"}:
        raise ValueError("unknown paired mode")
    a, b = outcomes(first), outcomes(second)
    if not a or set(a) != set(b):
        raise ValueError("paired scenarios differ/empty")
    differences = [int(b[key]) - int(a[key]) for key in sorted(a)]
    return {"mode": mode, "scenarios": len(a), "second_minus_first": sum(differences) / len(a),
            "ci95": cluster_interval(differences), "second_only_correct": differences.count(1),
            "first_only_correct": differences.count(-1), "note": "Caller must verify same dataset/protocol and valid run identities; exploratory paired CI."}

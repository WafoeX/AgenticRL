"""Recompute published final-test statistics from CSV; no model, GPU or raw replay.

This checks the released summary table, not the original model responses.
"""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

ARMS = ("SFT30", "GRPO20261011", "GRPO20261012", "GRPO20261013")
DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "results/research/final_test_feedback_review_v1/per_scenario.csv"
)


def summarize(path: Path) -> dict:
    """Require a complete, aligned 4 x 240 cohort before calculating anything."""
    groups = {arm: {} for arm in ARMS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            arm, sample_id = row["arm"], row["sample_id"]
            if arm not in groups or not sample_id or sample_id in groups[arm]:
                raise ValueError("unknown arm, empty ID or duplicate model/scenario")
            if row["correct"] not in ("0", "1"):
                raise ValueError("correct must be binary")
            if int(row["tool_calls"]) < 0:
                raise ValueError("negative tool calls")
            groups[arm][sample_id] = row
    # Keep the frozen task order preserved in the baseline CSV rows. Sorting IDs
    # changes index-to-scenario mapping for the finite seeded bootstrap draws.
    ids = list(groups[ARMS[0]])
    if len(ids) != 240 or any(set(group) != set(ids) for group in groups.values()):
        raise ValueError("expected four aligned cohorts of 240 scenarios")
    for sample_id in ids:
        metadata = {
            tuple(groups[arm][sample_id][key] for key in ("family", "depth", "generalization"))
            for arm in ARMS
        }
        if len(metadata) != 1:
            raise ValueError("scenario metadata mismatch across models")
    correct = {
        arm: [int(groups[arm][sample_id]["correct"]) for sample_id in ids]
        for arm in ARMS
    }
    rates = [statistics.mean(correct[arm]) for arm in ARMS]
    delta = [
        statistics.mean(correct[arm][i] for arm in ARMS[1:]) - correct[ARMS[0]][i]
        for i in range(240)
    ]
    rng = random.Random(20261011)
    boot = sorted(
        sum(delta[rng.randrange(240)] for _ in range(240)) / 240
        for _ in range(1000)
    )
    calls = {
        arm: sum(int(row["tool_calls"]) for row in groups[arm].values())
        for arm in ARMS
    }
    if calls[ARMS[0]] == 0:
        raise ValueError("zero baseline calls: relative call change undefined")
    return {
        "scope": "released CSV statistics only; not raw-response or model replay",
        "scenarios": 240,
        "model_scenario_results": 960,
        "answer_correct": {arm: sum(correct[arm]) for arm in ARMS},
        "accuracy_percent": {arm: 100 * rate for arm, rate in zip(ARMS, rates, strict=True)},
        "grpo_mean_percent": 100 * statistics.mean(rates[1:]),
        "grpo_sample_sd_pp": 100 * statistics.stdev(rates[1:]),
        "mean_gain_pp": 100 * statistics.mean(delta),
        "paired_bootstrap_ci95_pp": [100 * boot[24], 100 * boot[974]],
        "tool_calls": calls,
        "mean_tool_calls_change_percent": 100 * (
            statistics.mean(calls[arm] for arm in ARMS[1:]) / calls[ARMS[0]] - 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Read-only rollout diagnostics; no ART, reward, or training dependencies.

Infrastructure failures are identified solely by the existing ``failed == 1``
metric. Diagnostic valid-only statistics do NOT filter the training groups.
"""

from collections import defaultdict
from math import sqrt

SUCCESS_REWARD_THRESHOLD = 2.0


def summarize_trajectory_groups(groups) -> dict:
    """Observe groups without changing trajectories, rewards, or membership.

    ``avg_reward`` retains the legacy all-rollout denominator. Tool-use rates,
    ``valid_avg_reward``, and group std use only unflagged trajectories. Std is
    population std, matching the existing training diagnostic; groups with
    fewer than two unflagged trajectories are not counted in its denominator.
    ``model_failure_rate`` means reward <= 2 among unflagged rollouts, not a
    final-answer correctness judgment. Missing ``failed`` means unflagged for
    compatibility with older trajectories.
    """
    trajectories = []
    valid = []
    group_stds = []
    total_groups = 0
    for group in groups:
        total_groups += 1
        members = list(group.trajectories)
        trajectories.extend(members)
        group_valid = [t for t in members if t.metrics.get("failed", 0) != 1]
        valid.extend(group_valid)
        if len(group_valid) >= 2:
            rewards = [t.reward for t in group_valid]
            mean = sum(rewards) / len(rewards)
            group_stds.append(sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)))

    total = len(trajectories)
    valid_count = len(valid)
    infra_count = total - valid_count
    no_tool_count = sum(t.metrics.get("num_tools", 0) == 0 for t in valid)
    model_failures = sum(t.reward <= SUCCESS_REWARD_THRESHOLD for t in valid)
    return {
        "total_rollouts": total,
        "valid_trajectories": valid_count,
        "infra_failed_trajectories": infra_count,
        "avg_reward": sum(t.reward for t in trajectories) / total if total else 0.0,
        "valid_avg_reward": sum(t.reward for t in valid) / valid_count if valid_count else 0.0,
        "avg_tools_per_rollout": (
            sum(t.metrics.get("num_tools", 0) for t in valid) / valid_count
            if valid_count else 0.0
        ),
        "no_tool_rate": no_tool_count / valid_count if valid_count else 0.0,
        "model_failure_rate": model_failures / valid_count if valid_count else 0.0,
        "infra_failure_rate": infra_count / total if total else 0.0,
        "total_groups": total_groups,
        "groups_with_two_valid": len(group_stds),
        "groups_with_variance": sum(std > 0 for std in group_stds),
        "mean_group_std": sum(group_stds) / len(group_stds) if group_stds else 0.0,
    }


def format_diagnostics(stats: dict) -> str:
    """Render the diagnostic denominators explicitly (rates are fractions)."""
    lines = [
        f"  Diagnostics (read-only): rollouts={stats['total_rollouts']} "
        f"valid={stats['valid_trajectories']} infra failed={stats['infra_failed_trajectories']}",
        f"  Avg reward (all): {stats['avg_reward']:+.3f} | "
        f"Avg reward (valid): {stats['valid_avg_reward']:+.3f} | "
        f"Avg tools/rollout (valid): {stats['avg_tools_per_rollout']:.2f}",
        f"  No-tool rate (valid): {stats['no_tool_rate']:.1%} | "
        f"Model failure rate (valid, reward<=2): {stats['model_failure_rate']:.1%} | "
        f"Infra failure rate (all): {stats['infra_failure_rate']:.1%}",
        f"  Groups with variance (valid): {stats['groups_with_variance']}/"
        f"{stats['groups_with_two_valid']} eligible "
        f"({stats['total_groups']} total) | Mean group std (valid): {stats['mean_group_std']:.3f}",
    ]
    if stats["no_tool_rate"] > 0.6:
        lines.append("  WARNING: possible tool-call collapse")
    if not stats["valid_trajectories"]:
        lines.append("  No valid rollouts: valid-only metrics are unavailable (reported as 0).")
    return "\n".join(lines)


def summarize_by_scenario_type(scenarios: list[dict], groups) -> dict:
    """Aggregate stochastic rollout success by the existing scenario ``id``.

    Keep the evaluator's legacy reward > 2 threshold and all-attempt
    denominator. This is NOT deterministic task accuracy or pass@N. Missing
    ids are reported as ``unknown`` without rejecting old JSONL files.
    """
    groups = list(groups)
    if len(scenarios) != len(groups):
        raise ValueError("Each scenario must have exactly one evaluation group")
    buckets = defaultdict(list)
    for scenario, group in zip(scenarios, groups, strict=True):
        buckets[str(scenario.get("id") or "unknown")].append(group)

    result = {}
    for scenario_type, type_groups in sorted(buckets.items()):
        stats = summarize_trajectory_groups(type_groups)
        success_count = sum(
            t.reward > SUCCESS_REWARD_THRESHOLD
            for group in type_groups for t in group.trajectories
        )
        attempts = stats["total_rollouts"]
        result[scenario_type] = {
            "total_scenarios": len(type_groups),
            "total_attempts": attempts,
            "success_count": success_count,
            "accuracy_pct": success_count / attempts * 100 if attempts else 0.0,
            "avg_reward": stats["avg_reward"],
            "diagnostics": stats,
        }
    return result

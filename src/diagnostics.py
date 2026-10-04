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

    return {
        scenario_type: summarize_evaluation_groups(type_groups)
        for scenario_type, type_groups in sorted(buckets.items())
    }


def summarize_evaluation_groups(groups) -> dict:
    """Keep rollout success separate from task success (at least one success).

    With one deterministic attempt per scenario, ``task_success_pct`` is task
    accuracy. With k sampled attempts, it is observed pass@k. Flagged infra
    failures cannot count as task successes, but remain in legacy rollout
    reward/accuracy denominators. Missing attempts therefore make reported task
    success conservative; callers must also report the infrastructure rate.
    """
    groups = list(groups)
    stats = summarize_trajectory_groups(groups)
    success_count = sum(
        t.reward > SUCCESS_REWARD_THRESHOLD
        for group in groups for t in group.trajectories
    )
    task_success_count = sum(
        any(
            t.metrics.get("failed", 0) != 1 and t.reward > SUCCESS_REWARD_THRESHOLD
            for t in group.trajectories
        )
        for group in groups
    )
    attempts = stats["total_rollouts"]
    return {
        "total_scenarios": len(groups),
        "total_attempts": attempts,
        "success_count": success_count,
        "accuracy_pct": success_count / attempts * 100 if attempts else 0.0,
        "avg_reward": stats["avg_reward"],
        "task_success_count": task_success_count,
        "task_success_pct": task_success_count / len(groups) * 100 if groups else 0.0,
        "diagnostics": stats,
    }


def summarize_by_tool_depth(scenarios: list[dict], groups) -> dict:
    """Depth is the expected tool-call count, including repeated tool names."""
    depth_scenarios = [
        {"id": f"{len(scenario.get('expected_calls', []))}-step"}
        for scenario in scenarios
    ]
    return summarize_by_scenario_type(depth_scenarios, groups)


def summarize_complete_chains(scenarios, groups, outcomes) -> dict:
    """Independent complete-chain outcomes; never replace legacy reward stats."""
    groups = list(groups)
    if not (len(scenarios) == len(groups) == len(outcomes)):
        raise ValueError("Complete-chain outcomes must align with scenario groups")
    for group, flags in zip(groups, outcomes, strict=True):
        if len(group.trajectories) != len(flags):
            raise ValueError("Complete-chain outcomes must align with rollout attempts")

    def summarize(indices):
        attempts = successes = successful_scenarios = 0
        for index in indices:
            flags = [
                isinstance(flag, bool) and flag and t.metrics.get("failed", 0) != 1
                for t, flag in zip(groups[index].trajectories, outcomes[index], strict=True)
            ]
            attempts += len(flags)
            successes += sum(flags)
            successful_scenarios += any(flags)
        return {
            "total_scenarios": len(indices),
            "total_rollouts": attempts,
            "successful_rollouts": successes,
            "rollout_success_pct": successes / attempts * 100 if attempts else 0.0,
            "successful_scenarios": successful_scenarios,
            "task_success_pct": successful_scenarios / len(indices) * 100 if indices else 0.0,
        }

    types, depths = defaultdict(list), defaultdict(list)
    for index, scenario in enumerate(scenarios):
        types[str(scenario.get("id") or "unknown")].append(index)
        depths[f"{len(scenario.get('expected_calls', []))}-step"].append(index)
    return {
        "definition": "complete-tool-chain-v1; not final-answer correctness",
        "overall": summarize(list(range(len(groups)))),
        "by_scenario_type": {key: summarize(indices) for key, indices in sorted(types.items())},
        "by_tool_depth": {key: summarize(indices) for key, indices in sorted(depths.items())},
    }

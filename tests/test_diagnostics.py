"""Phase 1 regression checks: diagnostics never alter training data."""

from types import SimpleNamespace

import pytest

from src.diagnostics import (
    format_diagnostics,
    summarize_by_scenario_type,
    summarize_trajectory_groups,
)


def trajectory(reward, num_tools=0, failed=0):
    return SimpleNamespace(
        reward=reward, metrics={"num_tools": num_tools, "failed": failed},
    )


def group(*members):
    return SimpleNamespace(trajectories=list(members))


def test_infra_excluded_from_valid_diagnostics_but_not_mutated():
    members = [trajectory(4, 2), trajectory(-3, 0), trajectory(-3, 0, failed=1)]
    original_metrics = [dict(t.metrics) for t in members]
    original_group = group(*members)
    stats = summarize_trajectory_groups([original_group])
    assert stats["total_rollouts"] == 3
    assert stats["valid_trajectories"] == 2
    assert stats["infra_failed_trajectories"] == 1
    assert stats["avg_reward"] == pytest.approx(-2 / 3)  # Legacy denominator retained.
    assert stats["valid_avg_reward"] == 0.5
    assert stats["avg_tools_per_rollout"] == 1
    assert stats["no_tool_rate"] == 0.5
    assert stats["model_failure_rate"] == 0.5
    assert stats["infra_failure_rate"] == pytest.approx(1 / 3)
    assert stats["mean_group_std"] == 3.5
    assert original_group.trajectories == members
    assert all(a is b for a, b in zip(original_group.trajectories, members, strict=True))
    assert [t.metrics for t in members] == original_metrics
    assert [t.reward for t in members] == [4, -3, -3]


def test_real_model_no_tool_is_not_infrastructure_failure():
    stats = summarize_trajectory_groups([group(trajectory(-3), trajectory(-3))])
    assert stats["valid_trajectories"] == 2
    assert stats["no_tool_rate"] == 1
    assert stats["model_failure_rate"] == 1
    assert stats["infra_failure_rate"] == 0
    assert stats["groups_with_variance"] == 0
    assert stats["mean_group_std"] == 0
    assert "WARNING: possible tool-call collapse" in format_diagnostics(stats)


def test_group_std_uses_valid_members_and_eligible_group_denominator():
    groups = [
        group(trajectory(4, 1), trajectory(4, 1), trajectory(-3, failed=1)),
        group(trajectory(-3), trajectory(1, 1)),
        group(trajectory(4, 1), trajectory(-3, failed=1)),
        group(),
    ]
    stats = summarize_trajectory_groups(groups)
    assert stats["total_groups"] == 4
    assert stats["groups_with_two_valid"] == 2
    assert stats["groups_with_variance"] == 1
    assert stats["mean_group_std"] == 1  # population stds [0, 2]


@pytest.mark.parametrize("groups", [[], [group()], [group(trajectory(-3, failed=1))]])
def test_no_valid_rollouts_are_explicit(groups):
    stats = summarize_trajectory_groups(groups)
    assert stats["valid_trajectories"] == 0
    assert stats["no_tool_rate"] == 0
    assert stats["mean_group_std"] == 0
    text = format_diagnostics(stats)
    assert "No valid rollouts" in text
    assert "WARNING" not in text


def test_missing_failure_metric_remains_backward_compatible():
    old = SimpleNamespace(reward=4, metrics={"num_tools": 1})
    assert summarize_trajectory_groups([group(old)])["valid_trajectories"] == 1


def test_collapse_warning_threshold_is_strictly_above_sixty_percent():
    stats = summarize_trajectory_groups([
        group(*(trajectory(-3) for _ in range(3)), *(trajectory(4, 1) for _ in range(2))),
    ])
    assert stats["no_tool_rate"] == 0.6
    assert "WARNING" not in format_diagnostics(stats)


def test_task_type_aggregation_keeps_existing_success_denominator():
    scenarios = [{"id": "weather_compare"}, {"id": "simple_math"}, {"id": "weather_compare"}, {}]
    groups = [
        group(trajectory(4, 2), trajectory(-3)),
        group(trajectory(2, 1)),  # Exactly 2 is not success in the legacy evaluator.
        group(trajectory(4, 2), trajectory(-3, failed=1)),
        group(trajectory(4, 1)),
    ]
    report = summarize_by_scenario_type(scenarios, groups)
    weather = report["weather_compare"]
    assert weather["total_scenarios"] == 2
    assert weather["total_attempts"] == 4
    assert weather["success_count"] == 2
    assert weather["accuracy_pct"] == 50
    assert weather["diagnostics"]["infra_failed_trajectories"] == 1
    assert report["simple_math"]["accuracy_pct"] == 0
    assert report["unknown"]["accuracy_pct"] == 100


def test_task_type_aggregation_rejects_silent_alignment_loss():
    with pytest.raises(ValueError, match="exactly one"):
        summarize_by_scenario_type([{"id": "simple_math"}], [])


def test_all_existing_scenario_ids_are_preserved_without_prefix_guessing():
    ids = [
        "simple_math", "simple_weather", "simple_search", "simple_convert", "simple_lookup",
        "age_calculation", "weather_convert", "population_math", "search_and_convert",
        "weather_compare", "country_compare", "country_density_convert",
    ]
    report = summarize_by_scenario_type(
        [{"id": name} for name in ids], [group(trajectory(4, 1)) for _ in ids],
    )
    assert set(report) == set(ids)
    assert all(stats["accuracy_pct"] == 100 for stats in report.values())

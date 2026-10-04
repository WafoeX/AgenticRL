"""Phase 2 tests to run on the server; all inference/backend calls are mocked."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.diagnostics import summarize_by_tool_depth, summarize_evaluation_groups
from tests import test_phase1_integration as _phase1
from tests.test_rollout import TestRollout as _RolloutHelpers
from tests.test_rollout import rollout

# Reuse the isolated fixture without an imported name shadowed by test arguments.
entrypoints = _phase1.entrypoints
write_scenarios = _phase1.write_scenarios


def member(reward, failed=0):
    return SimpleNamespace(reward=reward, metrics={"num_tools": 1, "failed": failed})


def test_observed_pass_at_k_is_not_rollout_accuracy():
    groups = [
        SimpleNamespace(trajectories=[member(4), member(-3), member(-3), member(-3)]),
        SimpleNamespace(trajectories=[member(-3) for _ in range(4)]),
    ]
    stats = summarize_evaluation_groups(groups)
    assert stats["accuracy_pct"] == 12.5
    assert stats["task_success_pct"] == 50
    assert stats["task_success_count"] == 1


def test_depth_uses_expected_call_count_including_duplicates():
    scenarios = [
        {"expected_calls": [{"name": "get_weather"}]},
        {"expected_calls": [{"name": "get_weather"}, {"name": "get_weather"}]},
        {"expected_calls": [{"name": "lookup_db"}, {"name": "calculator"}, {"name": "convert_units"}]},
    ]
    groups = [SimpleNamespace(trajectories=[member(r)]) for r in (4, -3, 4)]
    stats = summarize_by_tool_depth(scenarios, groups)
    assert set(stats) == {"1-step", "2-step", "3-step"}
    assert stats["1-step"]["task_success_pct"] == 100
    assert stats["2-step"]["task_success_pct"] == 0
    assert stats["3-step"]["task_success_pct"] == 100


def test_flagged_infra_cannot_be_task_success_even_with_positive_reward():
    stats = summarize_evaluation_groups([SimpleNamespace(trajectories=[member(4, failed=1)])])
    assert stats["task_success_count"] == 0
    assert stats["diagnostics"]["infra_failure_rate"] == 1


@pytest.mark.parametrize("mode,generations,expected_rollouts,temperature,top_p", [
    ("deterministic", 8, 1, 0.0, 1.0),
    ("sample", 8, 8, 1.2, 0.95),
])
def test_eval_mode_options_and_separate_metric_names(
    entrypoints, tmp_path, mode, generations, expected_rollouts, temperature, top_p,
):
    path = write_scenarios(tmp_path)
    output = tmp_path / "report.json"
    results = asyncio.run(entrypoints.evaluate.main(
        path, generations, mode=mode, output_path=str(output),
    ))
    assert results["num_generations"] == expected_rollouts
    assert results["total_rollouts"] == expected_rollouts
    assert entrypoints.rollout_kwargs == [
        {"temperature": temperature, "top_p": top_p, "max_tokens": 512, "require_logprobs": False},
    ] * expected_rollouts
    if mode == "deterministic":
        assert results["deterministic_accuracy_pct"] == 100
        assert "pass_at_k_pct" not in results
    else:
        assert results["pass_at_k"] == 8
        assert results["pass_at_k_pct"] == 100
        assert results["accuracy_pct"] == 50
        assert "deterministic_accuracy_pct" not in results
    assert results["by_tool_depth"]["1-step"]["task_success_pct"] == 100
    assert json.loads(output.read_text())["mode"] == mode
    entrypoints.backend.train.assert_not_awaited()
    entrypoints.model.delete_checkpoints.assert_not_awaited()


def test_both_modes_use_one_checkpoint_registration(entrypoints, tmp_path):
    results = asyncio.run(entrypoints.evaluate.main(
        write_scenarios(tmp_path), 8, mode="both", output_path=str(tmp_path / "both.json"),
    ))
    assert set(results["evaluations"]) == {"deterministic", "sample"}
    assert results["deterministic_accuracy_pct"] == 100
    assert results["pass_at_k"] == 8
    assert results["evaluations"]["sample"]["total_rollouts"] == 8
    assert results["avg_reward"] == results["evaluations"]["sample"]["avg_reward"]
    assert len(entrypoints.rollout_kwargs) == 9
    entrypoints.model.register.assert_awaited_once()
    entrypoints.model.get_step.assert_awaited_once()
    entrypoints.backend.train.assert_not_awaited()


def test_four_scenario_batches_are_sequential_and_preserve_order(entrypoints, tmp_path):
    rows = [{"id": f"case_{i}", "task": f"task {i}", "expected_calls": []} for i in range(9)]
    path = tmp_path / "nine.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    active = 0
    maximum = 0
    seen = []

    async def tracked_rollout(model, scenario, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        seen.append(scenario["id"])
        await asyncio.sleep(0)
        active -= 1
        return member(4)

    entrypoints.evaluate.rollout = tracked_rollout
    results = asyncio.run(entrypoints.evaluate.main(
        str(path), 8, output_path=str(tmp_path / "batched.json"),
    ))
    assert [len(batch) for batch in entrypoints.gathered] == [4, 4, 1]
    assert maximum == 32  # Four scenarios times eight sampled attempts, not all 72.
    assert active == 0
    assert seen == [row["id"] for row in rows for _ in range(8)]
    assert [row["scenario_id"] for row in results["per_scenario"]] == [row["id"] for row in rows]


@pytest.mark.parametrize("kwargs", [
    {"batch_scenarios": 0}, {"num_generations": 0}, {"max_tokens": 0},
    {"mode": "unknown"}, {"mode": "deterministic", "deterministic_temperature": 0.5},
])
def test_invalid_eval_settings_fail_before_backend_registration(entrypoints, tmp_path, kwargs):
    settings = {"num_generations": 8, **kwargs}
    with pytest.raises(ValueError):
        asyncio.run(entrypoints.evaluate.main(write_scenarios(tmp_path), **settings))
    entrypoints.model.register.assert_not_awaited()


@pytest.mark.parametrize("missing", ["scenario", "attempt"])
def test_incomplete_gather_cannot_silently_change_denominators(entrypoints, tmp_path, missing):
    async def incomplete_gather(groups, **kwargs):
        resolved = list(groups)
        for group in resolved:
            group.trajectories = list(await asyncio.gather(*group.trajectories))
        if missing == "scenario":
            return []
        resolved[0].trajectories.pop()
        return resolved

    entrypoints.evaluate.art.gather_trajectory_groups = incomplete_gather
    output = tmp_path / "incomplete.json"
    with pytest.raises(RuntimeError, match="Evaluation returned"):
        asyncio.run(entrypoints.evaluate.main(
            write_scenarios(tmp_path), 8, output_path=str(output),
        ))
    assert not output.exists()


def test_empty_dataset_has_explicit_zero_counts_and_no_gather(entrypoints, tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    results = asyncio.run(entrypoints.evaluate.main(
        str(path), 8, output_path=str(tmp_path / "empty_report.json"),
    ))
    assert results["total_rollouts"] == 0
    assert results["total_scenarios"] == 0
    assert results["pass_at_k_pct"] == 0
    assert entrypoints.gathered == []


def test_old_yaml_evaluation_fields_default_and_cli_overrides(entrypoints, tmp_path, monkeypatch):
    config = tmp_path / "old.yaml"
    config.write_text("model:\n  name: old-model\nevaluation:\n  max_turns: 6\n")
    main = AsyncMock()
    monkeypatch.setattr(entrypoints.evaluate, "main", main)
    monkeypatch.setattr("sys.argv", ["eval", "--config", str(config)])
    entrypoints.evaluate.cli()
    call = main.await_args
    assert call is not None
    assert call.kwargs["mode"] == "sample"
    assert call.kwargs["batch_scenarios"] == 4
    assert call.kwargs["deterministic_temperature"] == 0
    assert call.kwargs["model_name"] == "old-model"
    config.write_text("evaluation:\n  batch_scenarios: 3\n  max_tokens: 128\n")
    monkeypatch.setattr("sys.argv", ["eval", "--config", str(config)])
    entrypoints.evaluate.cli()
    call = main.await_args
    assert call is not None
    assert call.kwargs["batch_scenarios"] == 3
    assert call.kwargs["max_tokens"] == 128
    monkeypatch.setattr("sys.argv", [
        "eval", "--config", str(config), "--batch-scenarios", "2", "--mode", "deterministic",
    ])
    entrypoints.evaluate.cli()
    call = main.await_args
    assert call is not None
    assert call.kwargs["batch_scenarios"] == 2
    assert call.kwargs["mode"] == "deterministic"


@pytest.mark.parametrize("options,temperature,top_p,logprobs", [
    ({}, 1.2, 0.95, True),
    ({"temperature": 0.0, "top_p": 1.0, "require_logprobs": False}, 0.0, 1.0, False),
    ({"temperature": 1.2, "top_p": 0.95, "require_logprobs": False}, 1.2, 0.95, False),
    ({"temperature": 0.7, "top_p": 0.8, "max_tokens": 256}, 0.7, 0.8, True),
])
def test_shared_rollout_defaults_raw_choices_and_eval_logprobs(options, temperature, top_p, logprobs):
    helpers = _RolloutHelpers()
    choices = [
        helpers._make_mock_choice("calculator", {"expression": "3+5"}),
        helpers._make_text_choice("8"),
    ]
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[choices[len(requests) - 1]])

    with patch("src.rollout.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        result = asyncio.run(rollout(MagicMock(), {
            "task": "What is 3+5?",
            "expected_calls": [{"name": "calculator", "arguments": {"expression": "3+5"}}],
        }, **options))
    assert result.reward == 4
    assert result.metrics["failed"] == 0
    assert result.metrics["num_tools"] == 1
    assert result.messages_and_choices[2] is choices[0]
    assert result.messages_and_choices[-1] is choices[1]
    assert len(requests) == 2
    for request in requests:
        assert request["temperature"] == temperature
        assert request["top_p"] == top_p
        assert request["max_tokens"] == options.get("max_tokens", 512)
        assert request["logprobs"] is logprobs
        if logprobs:
            assert request["top_logprobs"] == 1
        else:
            assert "top_logprobs" not in request


@pytest.mark.parametrize("infra_error,failed", [(False, 0), (True, 1)])
def test_shared_eval_loop_distinguishes_model_no_tool_and_infra(infra_error, failed):
    async def create(**kwargs):
        if infra_error:
            raise OSError("temporary server failure")
        return SimpleNamespace(choices=[_RolloutHelpers()._make_text_choice("no tools")])

    with patch("src.rollout.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        result = asyncio.run(rollout(MagicMock(), {
            "task": "What is 3+5?", "expected_calls": [],
        }, temperature=0.0, top_p=1.0, require_logprobs=False))
    assert result.reward == -3
    assert result.metrics["num_tools"] == 0
    assert result.metrics["failed"] == failed

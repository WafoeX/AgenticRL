"""Phase 3 server-run integration tests with inference and ART doubles only."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.diagnostics import summarize_complete_chains
from src.reward_v2 import is_complete_tool_chain
from src.rewards import compute_reward_by_version
from tests import test_phase1_integration as _phase1
from tests.test_rollout import TestRollout as _RolloutHelpers
from tests.test_rollout import rollout

entrypoints = _phase1.entrypoints
write_scenarios = _phase1.write_scenarios


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def trace(calls, results):
    return [{**copy.deepcopy(c), "result": r} for c, r in zip(calls, results, strict=True)]


@pytest.mark.parametrize("version,expected_reward", [("v1", 3.5), ("v2", 4.0)])
def test_real_shared_loop_sidecar_dynamic_reward_and_raw_choices(version, expected_reward):
    helpers = _RolloutHelpers()
    choices = [
        helpers._make_mock_choice("get_weather", {"city": "London"}),
        helpers._make_mock_choice("convert_units", {"value": 12, "from_unit": "celsius", "to_unit": "fahrenheit"}),
        helpers._make_text_choice("53.6 F"),
    ]
    requests = []
    sidecar = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[choices[len(requests) - 1]])

    with patch("src.rollout.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        trajectory = asyncio.run(rollout(MagicMock(), {
            "task": "London temperature in Fahrenheit?",
            "expected_calls": [call("get_weather", city="London"),
                               call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")],
        }, reward_version=version, trace_sink=sidecar))
    assert trajectory.reward == pytest.approx(expected_reward)
    assert trajectory.metrics["failed"] == 0
    assert trajectory.metrics["num_tools"] == 2
    assert json.loads(sidecar[0]["result"])["temperature"] == 12
    assert sidecar[1]["arguments"]["value"] == 12
    assert sidecar[1]["result"] == "53.6"
    assert trajectory.messages_and_choices[2] is choices[0]
    assert trajectory.messages_and_choices[4] is choices[1]
    assert trajectory.messages_and_choices[6] is choices[2]
    assert len(trajectory.messages_and_choices) == 7
    for request in requests:
        assert request["temperature"] == 1.2
        assert request["top_p"] == 0.95
        assert request["logprobs"] is True
        assert request["top_logprobs"] == 1
        assert "trace_sink" not in request
        assert "reward_version" not in request
    if version == "v1":
        assert set(trajectory.metrics) == {"num_tools", "num_turns", "failed"}
    else:
        assert trajectory.metrics["reward_v2/dynamic_score"] > 0
        assert trajectory.metrics["reward_v2/completion_score"] == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_one_choice_with_dependent_calls_does_not_receive_dynamic_credit(version):
    helpers = _RolloutHelpers()
    weather = helpers._make_mock_choice("get_weather", {"city": "London"})
    conversion = helpers._make_mock_choice("convert_units", {
        "value": 12, "from_unit": "celsius", "to_unit": "fahrenheit",
    })
    weather.message.tool_calls.append(conversion.message.tool_calls[0])
    choices = [weather, helpers._make_text_choice("done")]
    requests, sidecar = [], []
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[choices[len(requests) - 1]])

    with patch("src.rollout.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        trajectory = asyncio.run(rollout(MagicMock(), {
            "task": "convert London weather", "expected_calls": expected,
        }, reward_version=version, trace_sink=sidecar))
    assert len(requests) == 2
    assert [entry["turn"] for entry in sidecar] == [1, 1]
    predicted = [{"name": entry["name"], "arguments": entry["arguments"]} for entry in sidecar]
    assert not is_complete_tool_chain(predicted, expected, observations=sidecar)
    if version == "v2":
        assert trajectory.metrics["reward_v2/dynamic_score"] == 0
        assert trajectory.reward <= 2
    else:
        assert trajectory.reward == pytest.approx(3.5)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("infra", [False, True])
def test_no_tool_and_infra_remain_distinct_in_both_versions(version, infra):
    async def create(**kwargs):
        if infra:
            raise OSError("server transient")
        return SimpleNamespace(choices=[_RolloutHelpers()._make_text_choice("no tool")])

    sidecar = []
    with patch("src.rollout.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        trajectory = asyncio.run(rollout(MagicMock(), {
            "task": "calculate", "expected_calls": [call("calculator", expression="3+5")],
        }, reward_version=version, trace_sink=sidecar))
    assert trajectory.reward == -3
    assert trajectory.metrics["num_tools"] == 0
    assert trajectory.metrics["failed"] == int(infra)
    assert sidecar == []


def test_real_tool_numeric_string_error_cannot_be_complete():
    helpers = _RolloutHelpers()
    choices = [helpers._make_mock_choice("convert_units", {"value": "15", "from_unit": "kg", "to_unit": "lbs"}),
               helpers._make_text_choice("done")]
    requests = []
    sidecar = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[choices[len(requests) - 1]])

    with patch("src.rollout.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        trajectory = asyncio.run(rollout(MagicMock(), {
            "task": "convert", "expected_calls": [call("convert_units", value=15, from_unit="kg", to_unit="lbs")],
        }, reward_version="v2", trace_sink=sidecar))
    assert trajectory.metrics["failed"] == 0  # Tool returned an error, not a transport exception.
    assert sidecar[0]["result"].startswith("Error:")
    assert trajectory.metrics["reward_v2/argument_value_score"] > 0
    assert trajectory.metrics["reward_v2/completion_score"] == 0
    assert trajectory.reward <= 2


@pytest.mark.parametrize("kwargs", [{"reward_version": "typo"}, {"trace_sink": [{}]}])
def test_new_invalid_rollout_options_fail_before_inference(kwargs):
    with patch("src.rollout.AsyncOpenAI") as client:
        with pytest.raises(ValueError):
            asyncio.run(rollout(MagicMock(), {"task": "test"}, **kwargs))
        client.assert_not_called()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_train_and_validation_select_reward_without_filtering_or_regrouping(entrypoints, tmp_path, version):
    path = write_scenarios(tmp_path)
    asyncio.run(entrypoints.train.train(
        num_steps=5, rollouts_per_step=2, num_generations=2,
        train_scenarios_path=path, val_scenarios_path=path, reward_version=version,
    ))
    expected_kwargs = {} if version == "v1" else {"reward_version": "v2"}
    assert entrypoints.rollout_kwargs == [expected_kwargs] * 12
    assert entrypoints.backend.train.await_count == 5
    for backend_call, gathered in zip(entrypoints.backend.train.await_args_list, entrypoints.gathered[:5], strict=True):
        assert backend_call.args[1] is gathered
        assert [t.metrics["failed"] for t in gathered[0].trajectories] == [0, 1]
        assert backend_call.kwargs == {"learning_rate": 5e-5}


def test_reward_config_defaults_and_cli_override(entrypoints, tmp_path, monkeypatch):
    config = tmp_path / "reward.yaml"
    config.write_text(
        "model:\n  base: base\n  name: name\n  project: project\ntraining:\n"
        "  num_steps: 50\n  rollouts_per_step: 96\n  num_generations: 8\n"
        "  learning_rate: 0.00005\n  reward_version: v2\n"
    )
    train = AsyncMock()
    monkeypatch.setattr(entrypoints.train, "train", train)
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config)])
    entrypoints.train.main()
    assert train.await_args is not None
    assert train.await_args.kwargs["reward_version"] == "v2"
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config), "--reward-version", "v1"])
    entrypoints.train.main()
    assert train.await_args is not None
    assert train.await_args.kwargs["reward_version"] == "v1"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_eval_fixed_success_criterion_and_same_trace_reward_comparison(entrypoints, tmp_path, monkeypatch, version):
    scenarios = [
        {"id": "country_compare", "task": "compare", "expected_calls": [
            call("lookup_db", table="countries", key="France"), call("lookup_db", table="countries", key="Brazil")],
         "actual_calls": [call("lookup_db", table="countries", key="France")], "results": [{"population": 1000}]},
        {"id": "simple_math", "task": "3+5", "expected_calls": [call("calculator", expression="3+5")],
         "actual_calls": [call("calculator", expression="3 + 5")], "results": ["8"]},
    ]
    path = tmp_path / "audit.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in scenarios))
    requests = []

    async def fake_rollout(model, scenario, **kwargs):
        requests.append(kwargs)
        observations = trace(scenario["actual_calls"], scenario["results"])
        kwargs["trace_sink"].extend(observations)
        reward = compute_reward_by_version(
            scenario["actual_calls"], scenario["expected_calls"],
            version=kwargs.get("reward_version", "v1"), observations=observations,
        )
        return SimpleNamespace(reward=reward, metrics={"failed": 0, "num_tools": len(observations)})

    monkeypatch.setattr(entrypoints.evaluate, "rollout", fake_rollout)
    report = asyncio.run(entrypoints.evaluate.main(
        str(path), 2, reward_version=version, complete_chain_enabled=True,
        compare_rewards=True, save_traces=True, output_path=str(tmp_path / "audit_report.json"),
    ))
    assert len(requests) == 4  # Not a second generation pass for the second scorer.
    assert len({id(kwargs["trace_sink"]) for kwargs in requests}) == 4
    assert report["reward_version"] == version
    assert report["complete_chain_pass_at_k_pct"] == 50
    assert report["complete_chain"]["by_tool_depth"]["2-step"]["task_success_pct"] == 0
    assert report["complete_chain"]["by_scenario_type"]["simple_math"]["task_success_pct"] == 100
    assert report["reward_comparison"]["same_trajectories"] is True
    assert report["reward_comparison"]["available_rollouts"] == 4
    assert report["reward_comparison"]["v1_avg_reward"] == pytest.approx(2.5)
    assert report["per_scenario"][0]["attempts"][0]["complete_tool_chain"] is False
    assert report["per_scenario"][1]["attempts"][0]["complete_tool_chain"] is True
    assert report["per_scenario"][1]["attempts"][0]["tool_trace"][0]["arguments"]["expression"] == "3 + 5"
    if version == "v1":
        assert report["by_scenario_type"]["country_compare"]["task_success_pct"] == 100
        assert report["by_scenario_type"]["simple_math"]["task_success_pct"] == 0
    else:
        assert report["by_scenario_type"]["country_compare"]["task_success_pct"] == 0
        assert report["by_scenario_type"]["simple_math"]["task_success_pct"] == 100
    entrypoints.backend.train.assert_not_awaited()
    entrypoints.model.delete_checkpoints.assert_not_awaited()


def test_audit_follows_trajectory_identity_after_member_reorder(entrypoints, tmp_path, monkeypatch):
    counter = 0

    async def fake_rollout(model, scenario, **kwargs):
        nonlocal counter
        expression = "3+5" if counter == 0 else "4+5"
        counter += 1
        calls = [call("calculator", expression=expression)]
        observations = trace(calls, ["8" if expression == "3+5" else "9"])
        kwargs["trace_sink"].extend(observations)
        return SimpleNamespace(
            reward=compute_reward_by_version(calls, scenario["expected_calls"]),
            metrics={"failed": 0, "num_tools": 1},
        )

    async def reverse_gather(groups, **kwargs):
        groups = list(groups)
        for group in groups:
            group.trajectories = list(await asyncio.gather(*group.trajectories))
            group.trajectories.reverse()
        return groups

    monkeypatch.setattr(entrypoints.evaluate, "rollout", fake_rollout)
    monkeypatch.setattr(entrypoints.evaluate.art, "gather_trajectory_groups", reverse_gather)
    report = asyncio.run(entrypoints.evaluate.main(
        write_scenarios(tmp_path), 2, complete_chain_enabled=True, compare_rewards=True,
        save_traces=True, output_path=str(tmp_path / "reordered.json"),
    ))
    attempts = report["per_scenario"][0]["attempts"]
    assert [row["reward"] for row in attempts] == [2, 4]
    assert [row["tool_trace"][0]["arguments"]["expression"] for row in attempts] == ["4+5", "3+5"]
    assert [row["complete_tool_chain"] for row in attempts] == [False, True]
    assert [row["reward_comparison"]["v1"] for row in attempts] == [2, 4]


def test_failed_trace_cannot_count_as_success_or_comparison_data(entrypoints, tmp_path, monkeypatch):
    async def fake_rollout(model, scenario, **kwargs):
        calls = scenario["expected_calls"]
        kwargs["trace_sink"].extend(trace(calls, ["8"]))
        return SimpleNamespace(reward=4, metrics={"failed": 1, "num_tools": 1})

    monkeypatch.setattr(entrypoints.evaluate, "rollout", fake_rollout)
    report = asyncio.run(entrypoints.evaluate.main(
        write_scenarios(tmp_path), 1, mode="deterministic", complete_chain_enabled=True,
        compare_rewards=True, output_path=str(tmp_path / "failed.json"),
    ))
    assert report["complete_chain_accuracy_pct"] == 0
    assert report["reward_comparison"]["available_rollouts"] == 0
    assert report["reward_comparison"]["v1_avg_reward"] is None
    assert report["reward_comparison"]["excluded_rollouts"] == 1
    assert report["total_rollouts"] == 1  # No silent denominator shrink in legacy reports.


def test_complete_chain_summary_denominators_and_alignment():
    groups = [SimpleNamespace(trajectories=[
        SimpleNamespace(metrics={"failed": 0}), SimpleNamespace(metrics={"failed": 1})]),
        SimpleNamespace(trajectories=[SimpleNamespace(metrics={"failed": 0})])]
    scenarios = [{"id": "simple_math", "expected_calls": [{}]},
                 {"id": "country_compare", "expected_calls": [{}, {}]}]
    summary = summarize_complete_chains(scenarios, groups, [[True, True], [False]])
    assert summary["overall"]["successful_rollouts"] == 1
    assert summary["overall"]["rollout_success_pct"] == pytest.approx(100 / 3)
    assert summary["overall"]["task_success_pct"] == 50
    assert summary["by_tool_depth"]["2-step"]["task_success_pct"] == 0
    assert summarize_complete_chains(scenarios, groups, [[1, "yes"], [False]])["overall"]["successful_rollouts"] == 0
    with pytest.raises(ValueError, match="align"):
        summarize_complete_chains(scenarios, groups, [[True], [False]])
    with pytest.raises(ValueError, match="align"):
        summarize_complete_chains(scenarios, groups, [])


def test_eval_cli_opt_in_config_and_explicit_rollback(entrypoints, tmp_path, monkeypatch):
    config = tmp_path / "audit.yaml"
    config.write_text("training:\n  reward_version: v2\nevaluation:\n  complete_chain_enabled: true\n")
    main = AsyncMock()
    monkeypatch.setattr(entrypoints.evaluate, "main", main)
    monkeypatch.setattr("sys.argv", ["eval", "--config", str(config), "--compare-rewards", "--save-traces"])
    entrypoints.evaluate.cli()
    assert main.await_args is not None
    assert main.await_args.kwargs["reward_version"] == "v2"
    assert main.await_args.kwargs["complete_chain_enabled"] is True
    assert main.await_args.kwargs["compare_rewards"] is True
    assert main.await_args.kwargs["save_traces"] is True
    monkeypatch.setattr("sys.argv", ["eval", "--config", str(config), "--reward-version", "v1", "--no-complete-chain"])
    entrypoints.evaluate.cli()
    assert main.await_args is not None
    assert main.await_args.kwargs["reward_version"] == "v1"
    assert main.await_args.kwargs["complete_chain_enabled"] is False


def test_default_eval_retains_old_arguments_and_omits_opt_in_artifacts(entrypoints, tmp_path):
    report = asyncio.run(entrypoints.evaluate.main(
        write_scenarios(tmp_path), 2, output_path=str(tmp_path / "legacy.json"),
    ))
    assert all(set(kwargs) == {"temperature", "top_p", "max_tokens", "require_logprobs"}
               for kwargs in entrypoints.rollout_kwargs)
    assert "complete_chain" not in report
    assert "reward_comparison" not in report
    assert "attempts" not in report["per_scenario"][0]


def test_unknown_training_and_eval_reward_fails_before_registration(entrypoints, tmp_path):
    path = write_scenarios(tmp_path)
    with pytest.raises(ValueError, match="Unknown reward version"):
        asyncio.run(entrypoints.train.train(train_scenarios_path=path, val_scenarios_path=path, reward_version="v3"))
    with pytest.raises(ValueError, match="Unknown reward version"):
        asyncio.run(entrypoints.evaluate.main(path, 2, reward_version="v3"))
    entrypoints.model.register.assert_not_awaited()

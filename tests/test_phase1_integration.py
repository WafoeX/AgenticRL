"""Server-run mocked integration tests; no GPU or live ART server required."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def entrypoints(monkeypatch):
    """Load real entrypoints with isolated ART/rollout doubles, restored after use."""
    model = SimpleNamespace(
        register=AsyncMock(), get_step=AsyncMock(return_value=0),
        delete_checkpoints=AsyncMock(), log=AsyncMock(),
    )
    result = SimpleNamespace(metrics={"loss": 0.1}, step=1)
    backend = SimpleNamespace(train=AsyncMock(return_value=result))
    gathered = []
    rollout_kwargs = []
    call_count = 0

    async def fake_rollout(model, scenario, **kwargs):
        nonlocal call_count
        rollout_kwargs.append(kwargs)
        call_count += 1
        failed = int(call_count % 2 == 0)
        return SimpleNamespace(
            reward=-3.0 if failed else 4.0,
            metrics={"failed": failed, "num_tools": 0 if failed else 1},
        )

    class FakeGroup:
        def __init__(self, trajectories):
            self.trajectories = list(trajectories)

    async def gather(groups, **kwargs):
        groups = list(groups)
        for group in groups:
            group.trajectories = list(await asyncio.gather(*group.trajectories))
        gathered.append(groups)
        return groups

    class FakeLocalBackend:
        _monitor_openai_server = AsyncMock()

        def __new__(cls):
            return backend

    fake_art = ModuleType("art")
    fake_art.__dict__.update(
        TrainableModel=lambda **kwargs: model,
        TrajectoryGroup=FakeGroup,
        gather_trajectory_groups=gather,
    )
    fake_local = ModuleType("art.local")
    fake_backend = ModuleType("art.local.backend")
    fake_backend.__dict__["LocalBackend"] = FakeLocalBackend
    fake_rollout_module = ModuleType("src.rollout")
    fake_rollout_module.__dict__["rollout"] = fake_rollout
    for name, module in (
        ("art", fake_art), ("art.local", fake_local),
        ("art.local.backend", fake_backend), ("src.rollout", fake_rollout_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return SimpleNamespace(
        train=load("phase1_train_test", "src/train.py"),
        evaluate=load("phase1_eval_test", "scripts/eval_checkpoint.py"),
        model=model, backend=backend, gathered=gathered, rollout_kwargs=rollout_kwargs,
    )


def write_scenarios(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    path.write_text(json.dumps({
        "id": "simple_math", "task": "What is 3+5?",
        "expected_calls": [{"name": "calculator", "arguments": {"expression": "3+5"}}],
    }) + "\n")
    return str(path)


@pytest.mark.parametrize("enabled", [True, False])
def test_training_receives_identical_groups_including_infra_in_phase1(
    entrypoints, tmp_path, capsys, enabled,
):
    path = write_scenarios(tmp_path)
    asyncio.run(entrypoints.train.train(
        num_steps=1, rollouts_per_step=2, num_generations=2,
        train_scenarios_path=path, val_scenarios_path=path, diagnostics_enabled=enabled,
    ))
    call = entrypoints.backend.train.await_args
    assert call is not None
    assert call.args[1] is entrypoints.gathered[0]
    assert call.kwargs == {"learning_rate": 5e-5}
    assert [t.reward for t in call.args[1][0].trajectories] == [4, -3]
    assert [t.metrics["failed"] for t in call.args[1][0].trajectories] == [0, 1]
    assert entrypoints.rollout_kwargs == [{}, {}]  # No new sampling arguments.
    entrypoints.model.delete_checkpoints.assert_awaited_once()
    log_call = entrypoints.model.log.await_args
    assert log_call is not None
    assert log_call.args[0] is call.args[1]
    assert ("Diagnostics (read-only)" in capsys.readouterr().out) is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_eval_preserves_legacy_results_and_adds_only_optional_diagnostics(
    entrypoints, tmp_path, monkeypatch, enabled,
):
    path = write_scenarios(tmp_path)
    monkeypatch.chdir(tmp_path)
    asyncio.run(entrypoints.evaluate.main(path, 2, diagnostics_enabled=enabled))
    results = json.loads((tmp_path / "eval_results.json").read_text())
    assert results["total_rollouts"] == 2
    assert results["successes"] == 1
    assert results["accuracy_pct"] == 50
    assert results["avg_reward"] == 0.5
    assert results["per_scenario"][0]["total_attempts"] == 2
    assert entrypoints.rollout_kwargs == [{}, {}]
    entrypoints.backend.train.assert_not_awaited()
    if enabled:
        assert results["diagnostics"]["infra_failure_rate"] == 0.5
        assert results["by_scenario_type"]["simple_math"]["accuracy_pct"] == 50
        assert results["per_scenario"][0]["scenario_id"] == "simple_math"
    else:
        assert "diagnostics" not in results
        assert "by_scenario_type" not in results
        assert "accuracy_definition" not in results
        assert "scenario_id" not in results["per_scenario"][0]


def test_legacy_yaml_without_diagnostics_field_is_accepted(entrypoints, tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(
        "model:\n  base: old-base\n  name: old-name\n  project: old-project\n"
        "training:\n  num_steps: 50\n  rollouts_per_step: 96\n"
        "  num_generations: 8\n  learning_rate: 0.00005\n"
    )
    train = AsyncMock()
    monkeypatch.setattr(entrypoints.train, "train", train)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
    entrypoints.train.main()
    call = train.await_args
    assert call is not None
    assert call.kwargs["diagnostics_enabled"] is True
    assert call.kwargs["num_steps"] == 50
    assert call.kwargs["rollouts_per_step"] == 96


def test_no_diagnostics_cli_overrides_config(entrypoints, tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(
        "model:\n  base: old-base\n  name: old-name\n  project: old-project\n"
        "training:\n  num_steps: 50\n  rollouts_per_step: 96\n"
        "  num_generations: 8\n  learning_rate: 0.00005\n  diagnostics_enabled: true\n"
    )
    train = AsyncMock()
    monkeypatch.setattr(entrypoints.train, "train", train)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--no-diagnostics"])
    entrypoints.train.main()
    call = train.await_args
    assert call is not None
    assert call.kwargs["diagnostics_enabled"] is False

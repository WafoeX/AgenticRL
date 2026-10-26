"""Final-entry CPU contracts. Only validation-derived fixtures; never open test labels."""
import argparse
import asyncio
import copy
from types import SimpleNamespace

import pytest

import scripts.eval_research_final_test_v1 as final
from scripts.verify_research_eval_cpu import scripted_responses
from src.research_eval_protocol import load_development_records, plan_attempts
from src.research_evaluation import evaluate_one
from src.research_registration import load_registered_development
from src.research_schema import load_json


@pytest.fixture
def fixture_data():
    rows, _ = load_development_records(final.ROOT / "data/research_v1_frozen", limit=2)
    # Copy DEVELOPMENT examples to exercise only the new split routing.
    records = [{**copy.deepcopy(r), "split": "test"} for r in rows]
    protocol = load_json(final.ROOT / "configs/research/evaluation_v1.json")
    protocol["modes"] = ["deterministic"]
    return records, protocol, final.final_plan(records, protocol)


class FixtureBackend:
    def __init__(self, records, *, failure=False):
        self.responses = {r["sample_id"]: scripted_responses(r) for r in records}
        self.lookup = {}
        self.calls = 0
        self.closed = False
        self.failure = failure
        self.receipt = {"real_model_loaded": False, "fixture_only": True}
        self._torch = SimpleNamespace(cuda=SimpleNamespace(
            reset_peak_memory_stats=lambda _: None, max_memory_allocated=lambda _: 0,
            max_memory_reserved=lambda _: 0, empty_cache=lambda: None))

    async def complete(self, request):
        self.calls += 1
        if self.failure:
            raise RuntimeError("simulated OOM")
        key = self.lookup[request["attempt_id"]]
        return self.responses[key][request["turn"]]

    async def close(self):
        self.closed = True


def backend(records, plan, **kwargs):
    b = FixtureBackend(records, **kwargs)
    b.lookup = {s["attempt_id"]: s["sample_id"] for s in plan}
    return b


def test_preflight_bytes_only_no_model_or_labels(monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("must not parse labels or load weights")
    monkeypatch.setattr(final, "load_final_records", forbidden)
    monkeypatch.setattr(final, "resolve_identity", forbidden)
    result = asyncio.run(final.run(argparse.Namespace(run_gpu=False), loader=forbidden))
    assert result["state"] == "preflight-only"
    assert not result["test_labels_parsed"] and not result["GPU_started"]
    assert result["planned_tasks"] == 960


def test_old_entry_gates_remain_closed(fixture_data):
    records, protocol, _ = fixture_data
    with pytest.raises(ValueError, match="test"):
        plan_attempts(records, protocol)
    with pytest.raises(ValueError, match="test"):
        load_registered_development(final.ROOT, split="test")


def test_final_plan_never_disguises_split(fixture_data):
    records, protocol, plan = fixture_data
    assert all(p["split"] == "test" and p["mode"] == "deterministic" for p in plan)
    assert len({p["attempt_id"] for p in plan}) == 2
    with pytest.raises(ValueError):
        final.final_plan([{**records[0], "split": "validation"}], protocol)


def test_dual_save_reopen_tamper_duplicate_gap_rejected(tmp_path, fixture_data):
    records, protocol, plan = fixture_data
    b = backend(records, plan)
    row = asyncio.run(evaluate_one(records[0], plan[0], protocol, b))
    local, drive = tmp_path / "local", tmp_path / "drive"
    final.save(local / "attempts/0000.json", row, local, drive)
    assert final.read_prefix(local, plan, drive) == [row]
    asyncio.run(final.verify_rows(records, plan, protocol, [row]))
    bad = copy.deepcopy(row)
    bad["checker"]["answer"]["correct"] = not bad["checker"]["answer"]["correct"]
    with pytest.raises(ValueError, match="replay"):
        asyncio.run(final.verify_rows(records, plan, protocol, [bad]))
    (drive / "attempts/0000.json").write_text("{}")
    with pytest.raises(ValueError, match="differ"):
        final.read_prefix(local, plan, drive)
    (local / "attempts/0000.json").rename(local / "attempts/0002.json")
    with pytest.raises(ValueError, match="noncontiguous"):
        final.read_prefix(local, plan, drive)


@pytest.mark.parametrize("confirmation,balance,budget", [(None, "balance", 1000), (final.CONFIRM, "", 1000), (final.CONFIRM, "balance", 0)])
def test_missing_approval_fails_before_labels(confirmation, balance, budget, monkeypatch):
    monkeypatch.setattr(final, "preflight", lambda: ({}, {}, {}))
    args = argparse.Namespace(run_gpu=True, confirm=confirmation, balance_note=balance, wall_seconds=budget)
    with pytest.raises(ValueError, match="approval"):
        asyncio.run(final.run(args))


def setup_campaign(tmp_path, monkeypatch, fixture_data):
    records, protocol, plan = fixture_data
    cfg = {"models": [{"arm": arm, "step": 30, "checkpoint": arm, "fingerprint": arm}
                      for arm in ("SFT30", "GRPO20261011", "GRPO20261012", "GRPO20261013")],
           "base_fingerprint": "fixture"}
    monkeypatch.setattr(final, "preflight", lambda: (cfg, protocol, {"fixture_only": True}))
    monkeypatch.setattr(final, "load_final_records", lambda _: records)
    monkeypatch.setattr(final, "resolve_identity", lambda s: {"fingerprint": s["checkpoint_path"].split("/")[-1],
                                                             "base_fingerprint": "fixture"})
    monkeypatch.setattr(final, "verify_unchanged", lambda *_: None)
    args = argparse.Namespace(run_gpu=True, confirm=final.CONFIRM, balance_note="SIMULATED balance not GPU evidence",
                              wall_seconds=1000, output=str(tmp_path / "local"), backup=str(tmp_path / "drive"),
                              base_dir=str(tmp_path), drive_root=str(tmp_path), resume=False)
    return args, records, protocol, plan


def test_infra_stops_all_arms_preserves_failure_and_refuses_resume(tmp_path, monkeypatch, fixture_data):
    args, records, _, plan = setup_campaign(tmp_path, monkeypatch, fixture_data)
    b = backend(records, plan, failure=True)
    with pytest.raises(RuntimeError, match="infrastructure"):
        asyncio.run(final.run(args, loader=lambda *_: b))
    assert b.calls == 1 and b.closed
    assert load_json(tmp_path / "local/SFT30/attempts/0000.json")["infra_failure"]
    assert load_json(tmp_path / "local/manifest.json")["state"] == "failed-stop-investigate"
    args.resume = True
    with pytest.raises(ValueError, match="clean-boundary"):
        asyncio.run(final.run(args, loader=lambda *_: b))


def test_clean_pause_resume_replays_completed_prefix_no_double_count(tmp_path, monkeypatch, fixture_data):
    args, records, _, plan = setup_campaign(tmp_path, monkeypatch, fixture_data)
    args.wall_seconds = 421
    # Stop BEFORE any load at a clean boundary; no fake real model inference.
    tick = iter([0, 2, 2])
    monkeypatch.setattr(final, "time", SimpleNamespace(monotonic=lambda: next(tick)))
    result = asyncio.run(final.run(args, loader=lambda *_: pytest.fail("no load allowed")))
    assert result["state"] == "paused-clean-boundary"
    monkeypatch.setattr(final, "time", SimpleNamespace(monotonic=lambda: 0))
    # A tiny 2-row fixture intentionally cannot complete a 240-task arm.
    args.resume, args.wall_seconds = True, 1000
    b = backend(records, plan)
    result = asyncio.run(final.run(args, loader=lambda *_: b))
    assert result["state"] == "paused-clean-boundary" and b.closed
    saved = final.read_prefix(tmp_path / "local/SFT30", plan, tmp_path / "drive/SFT30")
    assert len(saved) == 2
    # Next resume verifies/replays the full prefix but generates no duplicate.
    calls = b.calls
    result = asyncio.run(final.run(args, loader=lambda *_: pytest.fail("prefix already complete")))
    assert b.calls == calls and result["state"] == "paused-clean-boundary"


def test_three_seed_paired_mean_not_720_independent():
    def rows(flags):
        return [{"sample_id": str(i), "mode": "deterministic", "checker": {"answer": {"correct": bool(v)}}}
                for i, v in enumerate(flags)]
    result = final.comparison([rows([0, 0, 1]), rows([1, 0, 1]), rows([1, 1, 1]), rows([0, 0, 1])])
    assert result["scenarios"] == 3
    assert result["mean_difference"] == pytest.approx(1 / 3)
    assert result["GRPO_sample_sd"] == pytest.approx(1 / 3)
    assert result["per_seed_paired"][0]["improved_ids"] == ["0"]

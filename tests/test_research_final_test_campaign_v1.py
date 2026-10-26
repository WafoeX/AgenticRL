"""960 SCRIPTED development-derived tasks; no real test labels/models/GPU."""
import argparse
import asyncio
import copy
from types import SimpleNamespace

import pytest

import scripts.eval_research_final_test_v1 as final
from scripts.verify_research_eval_cpu import scripted_responses
from src.research_eval_protocol import load_development_records
from src.research_schema import load_json


def test_complete_four_arms_and_no_double_generation_on_completed_resume(tmp_path, monkeypatch):
    source, _ = load_development_records(final.ROOT / "data/research_v1_frozen", limit=1)
    records = [{**copy.deepcopy(source[0]), "split": "test", "sample_id": f"fixture-{i}",
                "semantic_group_id": f"fixture-group-{i}"} for i in range(240)]
    protocol = load_json(final.ROOT / "configs/research/evaluation_v1.json")
    protocol["modes"] = ["deterministic"]
    cfg = {"models": [{"arm": arm, "step": 30, "checkpoint": arm, "fingerprint": arm}
                      for arm in ["SFT30", "GRPO20261011", "GRPO20261012", "GRPO20261013"]],
           "base_fingerprint": "fixture"}
    monkeypatch.setattr(final, "preflight", lambda: (cfg, protocol, {"fixture_only": True}))
    monkeypatch.setattr(final, "load_final_records", lambda _: records)
    monkeypatch.setattr(final, "resolve_identity", lambda s: {"fingerprint": s["checkpoint_path"].split("/")[-1],
                                                             "base_fingerprint": "fixture"})
    monkeypatch.setattr(final, "verify_unchanged", lambda *_: None)
    plan = final.final_plan(records, protocol)
    response = scripted_responses(records[0])
    loaded = []

    class Backend:
        def __init__(self):
            self.receipt = {"real_model_loaded": False, "SCRIPTED_only": True}
            self.closed, self.calls = False, 0
            self._torch = SimpleNamespace(cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda _: None, max_memory_allocated=lambda _: 0,
                max_memory_reserved=lambda _: 0, empty_cache=lambda: None))

        async def complete(self, request):
            self.calls += 1
            return response[request["turn"]]

        async def close(self):
            self.closed = True

    def loader(*_):
        assert not loaded or loaded[-1].closed, "must release previous model before next load"
        b = Backend()
        loaded.append(b)
        return b

    args = argparse.Namespace(run_gpu=True, confirm=final.CONFIRM, balance_note="SIMULATED CPU only",
                              wall_seconds=10000, output=str(tmp_path / "local"), backup=str(tmp_path / "drive"),
                              base_dir=str(tmp_path), drive_root=str(tmp_path), resume=False)
    result = asyncio.run(final.run(args, loader=loader))
    assert result["state"] == "completed" and len(loaded) == 4 and all(b.closed for b in loaded)
    for model in cfg["models"]:
        arm = model["arm"]
        assert len(final.read_prefix(tmp_path / "local" / arm, plan, tmp_path / "drive" / arm)) == 240
        summary = load_json(tmp_path / "local" / arm / "summary.json")
        assert summary["comparison_valid"]
        assert summary["evaluations"]["deterministic"]["overall"]["received_attempts"] == 240
    assert load_json(tmp_path / "local/comparison.json")["mean_difference"] == 0
    args.resume = True
    result = asyncio.run(final.run(args, loader=lambda *_: pytest.fail("completed run must not regenerate")))
    assert result["state"] == "completed" and len(loaded) == 4

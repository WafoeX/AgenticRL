"""Small CPU checks for the baseline evaluator; never load official weights."""
import argparse
import asyncio
import json
import zipfile
from types import SimpleNamespace

import pytest

from scripts import eval_research_baselines as baseline
from src.research_evaluation import RecordedBackend
from src.research_sft_run import SFTValidationBackend


def row(correct, calls, turns):
    return {"checker": {"answer": {"correct": correct}}, "tool_events": [{}] * calls, "requests": [{}] * turns}


def test_flow_keeps_failed_early_stop_separate():
    result = baseline.flow_metrics([row(True, 3, 4), row(False, 0, 1)])
    assert result["all_tasks"]["mean_tool_calls"] == 1.5
    assert result["answer_correct_tasks"]["mean_tool_calls"] == 3
    assert result["all_tasks"]["mean_turns"] == 2.5


def test_no_success_is_not_zero_call_efficiency():
    assert baseline.flow_metrics([row(False, 1, 2)])["answer_correct_tasks"]["mean_tool_calls"] is None


def test_same_decoder_for_both_arms():
    assert baseline.BaselineValidationBackend._generate is SFTValidationBackend._generate


class Model:
    def __init__(self, *, adapter=False, device="cuda:0", dtype="torch.bfloat16"):
        self.peft_config = {"default": {}} if adapter else {}
        self.active_adapters = ["default"] if adapter else []
        self.parameter = SimpleNamespace(device=device)
        self.weight = SimpleNamespace(dtype=dtype)

    def parameters(self):
        return [self.parameter]

    def named_parameters(self):
        return [("lora_A" if self.peft_config else "weight", self.parameter)]

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)


@pytest.mark.parametrize("kind,adapter", [("base", False), ("checkpoint", True)])
def test_backend_receipt_names_actual_arm(kind, adapter):
    backend = baseline.BaselineValidationBackend(Model(adapter=adapter), SimpleNamespace(chat_template="CPU fixture"),
                                                 None, {"kind": kind}, {})
    assert backend.receipt["active_adapters"] == (["default"] if adapter else [])


@pytest.mark.parametrize("kwargs", [{"adapter": True}, {"device": "cpu"}, {"dtype": "torch.float32"}])
def test_base_rejects_adapter_or_incomparable_precision(kwargs):
    with pytest.raises(ValueError):
        baseline.BaselineValidationBackend(Model(**kwargs), SimpleNamespace(chat_template="CPU fixture"),
                                            None, {"kind": "base"}, {})


def test_preflight_does_not_call_model_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "resolve_identity", lambda _: {"kind": "base", "fixture": True})
    def forbidden(*_):
        raise AssertionError("preflight must not load a model")
    args = argparse.Namespace(arm="B0", base_dir=str(tmp_path), checkpoint_dir=None,
                              output=str(tmp_path / "preflight"), run_gpu=False)
    result = asyncio.run(baseline.run(args, loader=forbidden))
    assert result["state"] == "preflight-only" and not result["training_performed"]
    assert len(json.loads((tmp_path / "preflight/plan.json").read_text())) == 180


@pytest.mark.parametrize("arm,checkpoint", [("B1", None), ("B0", "unexpected-adapter")])
def test_arm_path_requirements_precede_identity_resolution(tmp_path, arm, checkpoint):
    args = argparse.Namespace(arm=arm, base_dir=str(tmp_path), checkpoint_dir=checkpoint,
                              output=str(tmp_path / "unused"), run_gpu=False)
    with pytest.raises(ValueError):
        asyncio.run(baseline.run(args))
    assert not (tmp_path / "unused").exists()


@pytest.mark.skipif(
    not (baseline.ROOT / "results/20261013T055637Z-short-feedback.zip").is_file(),
    reason="optional historical raw feedback is not included in the public release",
)
def test_full_180_saved_SFT_replay_through_new_runner(tmp_path, monkeypatch):
    # Real historical responses, but NO new model inference or CUDA. Receipt
    # explicitly labels replay so these test artifacts cannot be model evidence.
    with zipfile.ZipFile(baseline.ROOT / "results/20261013T055637Z-short-feedback.zip") as z:
        manifest = json.loads(z.read("train/validation-10/manifest.json"))
        raw = [json.loads(line) for line in z.read("train/validation-10/attempts.jsonl").decode().splitlines()]
    monkeypatch.setattr(baseline, "resolve_identity", lambda _: manifest["identity"])
    monkeypatch.setattr(baseline, "verify_unchanged", lambda *_: None)
    receipt = {**manifest["loader_receipt"], "real_model_loaded": False, "replay_only": True}
    backend = RecordedBackend(receipt, raw)
    args = argparse.Namespace(arm="B1", base_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoint-10"),
                              output=str(tmp_path / "replay"), run_gpu=True)
    result = asyncio.run(baseline.run(args, loader=lambda *_: backend))
    summary = json.loads((tmp_path / "replay/summary.json").read_text())
    assert result["state"] == "completed" and backend.closed
    assert result["loader_receipt"]["real_model_loaded"] is False
    assert summary["comparison_valid"]
    assert summary["evaluations"]["deterministic"]["overall"]["deterministic_answer_accuracy"]["numerator"] == 74
    assert summary["tool_flow"]["all_tasks"]["tool_calls"] == 387
    assert summary["tool_flow"]["answer_correct_tasks"]["tool_calls"] == 171

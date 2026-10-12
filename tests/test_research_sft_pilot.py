"""Bounded pilot contracts. Tiny synthetic CPU tensors only, never pretrained training."""

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import pilot_research_sft as pilot_cli
from src.research_schema import load_json
from src.research_sft import pad_batch
from src.research_sft_pilot import (
    CONFIG,
    check_budget,
    select_records,
    validate_batch,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_exact():
    validate_config(load_json(ROOT / "configs/research/sft_pilot_v1.json"))


@pytest.mark.parametrize("key,value", [("updates", 3), ("batch_size", 4), ("seed", 1), ("max_length", 8192),
                                      ("dtype", "float32"), ("learning_rate", 0.001), ("max_wall_seconds", 3600),
                                      ("updates", 2.0), ("lora_r", True)])
def test_no_unbounded_override(key, value):
    config = copy.deepcopy(CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def fixture_records():
    return [{"split": "train", "reference_depth": d, "sample_id": f"s{i}"}
            for i, d in enumerate((2, 2, 1, 3, 1, 3))]


def test_depth_selection_is_fixed_not_by_correctness():
    selected = select_records(fixture_records())
    assert [r["sample_id"] for r in selected] == ["s2", "s0", "s3"]


@pytest.mark.parametrize("split", ["test", "validation"])
def test_never_select_holdout(split):
    rows = fixture_records()
    rows[-1]["split"] = split
    with pytest.raises(ValueError):
        select_records(rows)


def test_missing_depth():
    with pytest.raises(ValueError):
        select_records(fixture_records()[:3])


@pytest.mark.parametrize("remaining", [None, False, 0, -1, 21, float("nan"), float("inf"), "20"])
def test_explicit_budget_required(remaining):
    with pytest.raises(ValueError):
        check_budget(remaining, "2026-10-12T00:00:00Z")


def test_budget_counts_gpu_allocated_setup_time():
    now = datetime(2026, 10, 12, 0, 30, tzinfo=timezone.utc)
    receipt = check_budget(1, "2026-10-12T00:00:00Z", now=now)
    assert receipt["allocated_seconds_before_pilot"] == 1800
    with pytest.raises(ValueError, match="insufficient"):
        check_budget(0.5, "2026-10-12T00:00:00Z", now=now)
    with pytest.raises(ValueError):
        check_budget(1, "2026-10-12T01:00:00Z", now=now)
    with pytest.raises(ValueError, match="timezone"):
        check_budget(1, "2026-10-12T00:00:00", now=now)


def batch_fixture():
    rows = [{"input_ids": [1, 2, 3][:n], "attention_mask": [1] * n,
             "labels": [-100, 2, 3][:n], "token_count": n} for n in (2, 3, 3)]
    return rows, pad_batch(rows, pad_token_id=0)


def test_mask_preserved():
    rows, batch = batch_fixture()
    assert validate_batch(batch, rows) == 5


@pytest.mark.parametrize("kind", ["context", "padding", "labels", "empty", "attention", "cardinality"])
def test_reject_collator_corruption(kind):
    rows, batch = batch_fixture()
    if kind == "context":
        batch["labels"][0][0] = 1
    elif kind == "padding":
        batch["labels"][0][-1] = 0
    elif kind == "labels":
        batch["labels"][1][1] = 9
    elif kind == "empty":
        rows[0]["labels"] = [-100, -100]
        batch["labels"][0] = [-100, -100, -100]
    elif kind == "attention":
        batch["attention_mask"][0][-1] = 1
    else:
        batch["labels"].pop()
    with pytest.raises(ValueError):
        validate_batch(batch, rows)


def test_gpu_gate_before_dependency_or_output(tmp_path):
    target = tmp_path / "run"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_research_sft.py"),
                             "--base-dir", str(tmp_path / "missing"), "--output", str(target), "--run-gpu"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "explicit pilot-only" in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("legacy", [[], ["--gpu-allocated-at", "2099-01-01T00:00:00Z"],
                                   ["--remaining-gpu-hours", "0", "--gpu-allocated-at", "not-a-date"]])
def test_gpu_entry_needs_no_manual_budget_or_timestamp(monkeypatch, tmp_path, legacy):
    target = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["pilot", "--base-dir", str(tmp_path / "base"),
                                    "--output", str(target), "--run-gpu", "--confirm-pilot-only", *legacy])
    monkeypatch.setattr(pilot_cli.subprocess, "check_output",
                        lambda command, **kwargs: "" if "status" in command else "fixture-commit")

    def reached_prepare(*args):
        raise RuntimeError("preflight reached without manual budget gate")

    monkeypatch.setattr(pilot_cli, "prepare", reached_prepare)
    with pytest.raises(RuntimeError, match="preflight reached"):
        pilot_cli.main()
    result = json.loads((target / "result.json").read_text())
    assert result["budget"]["total_GPU_budget_verified"] is False
    assert result["budget"]["actual_allocation_time"] is None
    assert result["actual_updates"] == 0
    assert result["pilot_process_wall_hours"] >= 0
    assert "allocated_GPU_hours_to_end" not in result


def test_tiny_cpu_optimizer_loss_and_failure_contracts():
    python = Path(os.environ.get("RESEARCH_PILOT_PYTHON", str(Path.home() / ".cache/agenticrl-pilot-venv/bin/python")))
    if not python.exists():
        if "RESEARCH_PILOT_PYTHON" in os.environ:
            pytest.fail("explicit CPU contract-test Python unavailable")
        pytest.skip("optional local torch contract environment unavailable")
    code = '''
import copy, time
from types import SimpleNamespace
import torch
from src.research_sft_pilot import CONFIG, train_updates, tensor_fingerprint
class Tiny(torch.nn.Module):
    def __init__(self, mode='normal'):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.arange(25, dtype=torch.float32).reshape(5,5)/25)
        self.mode = mode
    def forward(self, input_ids, attention_mask, labels, use_cache=False):
        logits = self.lora_A[input_ids]
        target = labels[:,1:]
        if self.mode == 'unmasked': target = input_ids[:,1:]
        loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,5), target.reshape(-1), ignore_index=-100)
        if self.mode == 'nan': loss = loss * float('nan')
        if self.mode == 'zero': loss = loss * 0
        return SimpleNamespace(loss=loss, logits=logits)
batch = {'input_ids': torch.tensor([[1,2,3],[2,3,4],[3,4,1]]),
         'labels': torch.tensor([[-100,-100,3],[-100,3,4],[-100,-100,1]]),
         'attention_mask': torch.ones(3,3,dtype=torch.long)}
original = {k:v.clone() for k,v in batch.items()}
logs = []
m = Tiny()
receipt = train_updates(m,batch,CONFIG,torch,deadline=time.monotonic()+30,emit=logs.append)
assert receipt['actual_updates'] == 2 and len(logs) == 2 and receipt['max_parameter_delta'] > 0
assert receipt['initial_adapter_sha256'] != receipt['final_adapter_sha256']
assert all(row['shifted_supervised_tokens'] == 4 and row['grad_norm_before_clip'] > 0
           and row['changed_tensors_this_update'] > 0 for row in logs)
assert all(torch.equal(v, original[k]) for k,v in batch.items())
assert tensor_fingerprint({'x':torch.tensor([1.,2.])},torch) != tensor_fingerprint({'x':torch.tensor([1.,3.])},torch)
for mode in ('unmasked','nan','zero'):
    try: train_updates(Tiny(mode),batch,CONFIG,torch,deadline=time.monotonic()+30,emit=lambda r:None)
    except ValueError: pass
    else: raise AssertionError(mode+' accepted')
empty = copy.deepcopy(batch)
empty['labels'].fill_(-100)
try: train_updates(Tiny(),empty,CONFIG,torch,deadline=time.monotonic()+30,emit=lambda r:None)
except ValueError: pass
else: raise AssertionError('empty supervision accepted')
try: train_updates(Tiny(),batch,CONFIG,torch,deadline=time.monotonic()-1,emit=lambda r:None)
except TimeoutError: pass
else: raise AssertionError('deadline ignored')
m = Tiny()
m.lora_A.requires_grad_(False)
try: train_updates(m,batch,CONFIG,torch,deadline=time.monotonic()+30,emit=lambda r:None)
except ValueError: pass
else: raise AssertionError('zero trainable params accepted')
# Exercise real Transformers/PEFT APIs on a RANDOM tiny Qwen, not official weights.
import tempfile
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
torch.manual_seed(7)
base_config = Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         attention_dropout=0.0)
base = Qwen2ForCausalLM(base_config)
base_state = {n:p.detach().clone() for n,p in base.state_dict().items()}
model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
                       lora_dropout=0.0, target_modules=CONFIG['target_modules'], bias='none'))
api_logs = []
trained = train_updates(model,batch,CONFIG,torch,deadline=time.monotonic()+30,emit=api_logs.append)
assert trained['actual_updates'] == 2 and trained['max_parameter_delta'] > 0
model.eval()
with torch.no_grad(): expected = model(input_ids=batch['input_ids'],use_cache=False,logits_to_keep=1).logits
with tempfile.TemporaryDirectory() as directory:
    model.save_pretrained(directory,safe_serialization=True,save_embedding_layers=False)
    fresh = Qwen2ForCausalLM(base_config)
    fresh.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(fresh,directory,is_trainable=False)
    reloaded.eval()
    with torch.no_grad(): actual = reloaded(input_ids=batch['input_ids'],use_cache=False,logits_to_keep=1).logits
    assert torch.equal(expected,actual)
    state = {n:p for n,p in model.named_parameters() if 'lora_' in n}
    loaded = {n:p for n,p in reloaded.named_parameters() if 'lora_' in n}
    assert tensor_fingerprint(state,torch) == tensor_fingerprint(loaded,torch)
print('tiny CPU contracts passed; no pretrained model or GPU')
'''
    result = subprocess.run([str(python), "-c", code], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "contracts passed" in result.stdout

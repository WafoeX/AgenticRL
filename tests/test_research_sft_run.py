"""R3.5/3.6 CPU contracts. No pretrained optimization/inference/GPU."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import train_research_sft as cli
from scripts.verify_research_eval_cpu import create_fixture_spec
from src.research_model_identity import artifact_inventory, file_hash, resolve_identity
from src.research_schema import load_json
from src.research_sft_run import (
    CONFIG_PATH,
    batch_indices,
    select_checkpoint,
    stage_grpo_starts,
    validate_config,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_json(ROOT / CONFIG_PATH)


def test_first_contract(config):
    validate_config(config)


@pytest.mark.parametrize("key,value", [("learning_rate", 1e-3), ("seed", 12), ("test_allowed", True),
    ("gradient_accumulation", 3), ("micro_batch_size", 3.0), ("automatic_retries", 1), ("dtype", "float32"),
    ("terminal_step", 100), ("candidate_steps", [10]), ("validation_scenarios", 3), ("max_length", 512)])
def test_no_silent_contract_change(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_versioned_extension(config):
    new = {**config, "version": "research-sft-short-v2", "terminal_step": 20, "candidate_steps": [15, 20]}
    validate_config(new, parent=config, start_step=10)
    with pytest.raises(ValueError):
        validate_config(new)
    new["version"] = config["version"]
    with pytest.raises(ValueError):
        validate_config(new, parent=config, start_step=10)


@pytest.mark.parametrize("steps", [[], [10, 20], [20, 15], [15, 15, 20], [15], [15.0, 20], [True, 20], [21]])
def test_bad_extension_steps(config, steps):
    new = {**config, "version": "research-sft-short-v2", "terminal_step": 20, "candidate_steps": steps}
    with pytest.raises(ValueError):
        validate_config(new, parent=config, start_step=10)


def test_shuffled_cursor_no_repetition_or_drop(config):
    groups = [batch_indices(1200, step, config) for step in range(1, 101)]
    assert all(len(group) == 12 for group in groups)
    assert sorted(i for group in groups for i in group) == list(range(1200))
    assert groups[:10] == [batch_indices(1200, step, copy.deepcopy(config)) for step in range(1, 11)]
    assert batch_indices(1200, 101, config) != groups[0]
    with pytest.raises(ValueError):
        batch_indices(1199, 1, config)


def candidates():
    return [{"step": step, "scenarios": 180, "comparison_valid": True, "answer_correct": a, "dynamic_valid": d}
            for step, a, d in [(5, 100, 40), (10, 100, 41)]]


def test_lexicographic_selection():
    rows = candidates()
    assert select_checkpoint(rows, [5, 10])["step"] == 10
    rows[0]["answer_correct"] = 101
    assert select_checkpoint(rows, [5, 10])["step"] == 5
    rows[0]["answer_correct"] = 100
    rows[0]["dynamic_valid"] = 41
    assert select_checkpoint(rows, [5, 10])["step"] == 5
    for row in rows:
        row.update(answer_correct=0, dynamic_valid=0)
    assert select_checkpoint(rows, [5, 10])["step"] == 5  # zero accuracy is not infra


@pytest.mark.parametrize("kind", ["infra", "missing", "extra", "duplicate", "subset", "bad_count", "bool"])
def test_no_invalid_partial_selection(kind):
    rows = candidates()
    if kind == "infra":
        rows[1]["comparison_valid"] = False
    elif kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append({**rows[0], "step": 20})
    elif kind == "duplicate":
        rows[1]["step"] = 5
    elif kind == "subset":
        rows[1]["scenarios"] = 3
    elif kind == "bad_count":
        rows[0]["answer_correct"] = 181
    else:
        rows[0]["dynamic_valid"] = True
    with pytest.raises(ValueError):
        select_checkpoint(rows, [5, 10])


def test_grpo_byte_copies_not_links_or_optimizer(tmp_path):
    spec = create_fixture_spec(tmp_path / "fixture", checkpoint=True)
    origin = resolve_identity(spec)
    state = Path(spec["checkpoint_path"]) / "continuation-state"
    state.mkdir()
    (state / "state.safetensors").write_bytes(b"not-a-real-optimizer")
    receipt = stage_grpo_starts(spec, tmp_path / "starts", selection={"step": 7})
    a = receipt["arms"]["G1"]["identity"]
    b = receipt["arms"]["G2"]["identity"]
    assert a["fingerprint"] == b["fingerprint"] == origin["fingerprint"]
    assert a["checkpoint_path"] != b["checkpoint_path"]
    for directory in (a["checkpoint_path"], b["checkpoint_path"]):
        assert not (Path(directory) / "continuation-state").exists()
        assert not (Path(directory) / ".art").exists()
        assert all(not p.is_symlink() for p in Path(directory).iterdir())
    (Path(a["checkpoint_path"]) / "adapter_model.safetensors").write_bytes(b"mutate-G1")
    assert artifact_inventory(Path(b["checkpoint_path"])) == origin["checkpoint_files"]
    assert resolve_identity(spec) == origin
    with pytest.raises(FileExistsError):
        stage_grpo_starts(spec, tmp_path / "starts", selection={"step": 7})
    with pytest.raises(ValueError):
        stage_grpo_starts(spec, tmp_path / "wrong", selection={"step": 2})


def committed_fixture(tmp_path, config):
    spec = create_fixture_spec(tmp_path, checkpoint=True)
    checkpoint = Path(spec["checkpoint_path"])
    identity = resolve_identity(spec)
    marker = {"version": "research-sft-continuation-v1", "checkpoint_path": str(checkpoint), "step": 7,
              "config": config, "identity": identity, "model_spec": spec,
              "files": {p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()}}
    write_json(tmp_path / "continuation-7.json", marker)
    write_json(tmp_path / "result.json", {"last_committed_step": 7, "last_checkpoint": str(checkpoint)})
    return checkpoint


def test_last_committed_even_if_validation_failed(tmp_path, config):
    path = committed_fixture(tmp_path, config)
    assert cli.load_resume_source(path)["step"] == 7
    write_json(tmp_path / "result.json", {"state": "failed", "last_committed_step": 7, "last_checkpoint": str(path)})
    assert cli.load_resume_source(path)["step"] == 7
    write_json(tmp_path / "result.json", {"last_committed_step": 10, "last_checkpoint": str(tmp_path / "checkpoint-10")})
    with pytest.raises(ValueError, match="last committed"):
        cli.load_resume_source(path)


@pytest.mark.parametrize("change", ["weight", "extra", "missing", "state"])
def test_resume_rejects_corruption(tmp_path, config, change):
    path = committed_fixture(tmp_path, config)
    if change == "weight":
        (path / "adapter_model.safetensors").write_bytes(b"changed")
    elif change == "extra":
        (path / "unexpected.json").write_text("{}")
    elif change == "missing":
        (path / "adapter_config.json").unlink()
    else:
        write_json(tmp_path / "result.json", {"last_committed_step": 7, "last_checkpoint": "some-other-run"})
    with pytest.raises(ValueError):
        cli.load_resume_source(path)


def test_gpu_explicit_gate_before_output(tmp_path):
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/train_research_sft.py"),
        "--base-dir", str(tmp_path / "missing"), "--output", str(output), "--run-gpu"],
        cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0 and "explicit --confirm-short-sft" in completed.stderr
    assert not output.exists()


def test_failure_ledger_before_model_load(monkeypatch, tmp_path):
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--base-dir", str(tmp_path / "base"), "--output", str(output)])

    def fail(*args):
        raise ValueError("preflight rejected")

    monkeypatch.setattr(cli, "prepare", fail)
    with pytest.raises(ValueError, match="preflight rejected"):
        cli.main()
    result = load_json(output / "result.json")
    assert result["state"] == "failed" and result["actual_updates_this_run"] == 0
    assert not result["training_performed"] and result["process_wall_hours"] >= 0


def test_tiny_cpu_accumulation_and_exact_resume(tmp_path):
    python = Path(os.environ.get("RESEARCH_PILOT_PYTHON", str(Path.home() / ".cache/agenticrl-pilot-venv/bin/python")))
    if not python.exists():
        pytest.skip("optional tiny torch CPU contract environment unavailable")
    code = r'''
import copy, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from src.research_sft_run import optimizer_update, save_resume, restore_resume
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.arange(25).float().reshape(5,5)/25)
    def forward(self,input_ids,attention_mask,labels,use_cache):
        logits = self.lora_A[input_ids]
        loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,5),labels[:,1:].reshape(-1),ignore_index=-100)
        return SimpleNamespace(loss=loss,logits=logits)
rows = [{'input_ids':[1,2,3][:n], 'labels':[-100,2,3][:n], 'attention_mask':[1]*n} for n in (2,3,3,2)]
cfg = dict(micro_batch_size=1,gradient_accumulation=4,max_grad_norm=1.)
model = Tiny(); other=copy.deepcopy(model)
opt = torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=0.)
opt2 = torch.optim.AdamW(other.parameters(),lr=.0001,weight_decay=0.)
a = optimizer_update(model,opt,rows,cfg,torch,pad_id=0,device='cpu')
b = optimizer_update(other,opt2,rows,{**cfg,'micro_batch_size':4,'gradient_accumulation':1},torch,pad_id=0,device='cpu')
assert abs(a['loss']-b['loss']) < 1e-7
assert torch.allclose(model.lora_A,other.lora_A,atol=1e-8,rtol=0)
assert a['shifted_supervised_tokens']==6 and a['changed_tensors_this_update']==1
state=Path(sys.argv[1])/'state'
contract={'fixture':'not-pretrained-not-GPU'}
save_resume(state,model,opt,torch,step=1,contract=contract,cuda=False)
expected_random=torch.rand(3)
cloned=copy.deepcopy(model)
newopt=torch.optim.AdamW(cloned.parameters(),lr=.0001,weight_decay=0.)
restore_resume(state,cloned,newopt,torch,expected_contract=contract,device='cpu')
assert torch.equal(expected_random,torch.rand(3))
optimizer_update(model,opt,rows,cfg,torch,pad_id=0,device='cpu')
optimizer_update(cloned,newopt,rows,cfg,torch,pad_id=0,device='cpu')
assert torch.equal(model.lora_A,cloned.lora_A)
for key in ('step','exp_avg','exp_avg_sq'):
 assert torch.equal(opt.state_dict()['state'][0][key],newopt.state_dict()['state'][0][key])
try:
 restore_resume(state,cloned,newopt,torch,expected_contract={'fixture':'changed'},device='cpu')
 raise AssertionError('accepted changed resume contract')
except ValueError: pass
bad=[{**r,'labels':[-100]*len(r['input_ids'])} for r in rows]
try:
 optimizer_update(model,opt,bad,cfg,torch,pad_id=0,device='cpu')
 raise AssertionError('accepted empty loss')
except ValueError: pass
print('tiny CPU token-weighted update and exact optimizer/RNG continuation passed')
'''
    completed = subprocess.run([str(python), "-c", code, str(tmp_path)], cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "exact optimizer/RNG continuation passed" in completed.stdout


def test_full_validation_entry_with_cpu_fixtures(tmp_path):
    python = Path(os.environ.get("RESEARCH_PILOT_PYTHON", str(Path.home() / ".cache/agenticrl-pilot-venv/bin/python")))
    if not python.exists():
        pytest.skip("optional tiny torch CPU environment unavailable")
    code = r'''
import asyncio, copy, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts import train_research_sft as entry
from scripts import verify_research_eval_cpu as fixture
from src.research_eval_protocol import plan_attempts
from src.research_registration import load_registered_development
from src.research_model_identity import resolve_identity
from src.research_schema import load_json
root=Path.cwd(); output=Path(sys.argv[1])
spec=fixture.create_fixture_spec(output/'fake-model',checkpoint=True)
identity=resolve_identity(spec)
records,_=load_registered_development(root,split='validation')
protocol=load_json(root/'configs/research/evaluation_v1.json'); protocol['modes']=['deterministic']
plan=plan_attempts(records,protocol)
responses={p['attempt_id']:fixture.scripted_responses(records[p['scenario_index']]) for p in plan}
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__(); self.lora_A=torch.nn.Parameter(torch.ones(5,5)); self.active_adapters=['default']
 def forward(self,input_ids,use_cache): return SimpleNamespace(logits=self.lora_A[input_ids])
model=Tiny()
entry.load_base=lambda *args: copy.deepcopy(model)
original_import=entry.importlib.import_module
def imported(name):
 if name=='peft': return SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda base,*args,**kw:base))
 return original_import(name)
entry.importlib.import_module=imported
class TorchProxy:
 cuda=SimpleNamespace(empty_cache=lambda:None)
 def __getattr__(self,name): return getattr(torch,name)
 def tensor(self,*args,**kw): kw['device']='cpu'; return torch.tensor(*args,**kw)
proxy=TorchProxy()
entry.SFTValidationBackend=lambda *args: fixture.FixtureBackend(identity,responses)
config=load_json(root/'configs/research/sft_short_v1.json')
result=asyncio.run(entry.validate_candidate(output/'validation',spec,identity,model,
 SimpleNamespace(bos_token_id=1,eos_token_id=2),records,protocol,{},proxy,config))
assert result['scenarios']==180 and result['answer_correct']==180
assert result['dynamic_valid']==106 and result['comparison_valid']
assert len((output/'validation'/'attempts.jsonl').read_text().splitlines())==180
assert len((output/'validation'/'ledger.jsonl').read_text().splitlines())==180
summary=load_json(output/'validation'/'summary.json')
assert summary['comparison_valid']
print('full 180 validation fixture entry passed; not GPU/model capability')
'''
    completed = subprocess.run([str(python), "-c", code, str(tmp_path)], cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "full 180 validation fixture entry passed" in completed.stdout

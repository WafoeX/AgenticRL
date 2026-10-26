"""CPU-only real autograd/AdamW tests; never load an official model."""
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import pilot_research_grpo_step30 as entry
from scripts.research_grpo_native import (
    PILOT,
    clipped_loss,
    group_advantages,
    optimizer_update,
    rollout,
    token_logps,
    validate_config,
)
from src.research_checker import protocol_plan
from src.research_eval_backend import PromptBudgetExceeded
from src.research_grpo_reward import score
from src.research_registration import load_registered_development
from src.research_sft_pilot import tensor_fingerprint
from src.research_sft_run import lora_parameters, restore_resume, save_resume

ROOT = Path(__file__).resolve().parents[1]


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            self.embedding = torch.nn.Embedding(7, 4)
            self.head = torch.nn.Linear(4, 7, bias=False)
            self.embedding.weight.requires_grad_(False)
            self.head.weight.requires_grad_(False)
            self.lora_A = torch.nn.Parameter(torch.randn(4, 2) * .1)
            self.lora_B = torch.nn.Parameter(torch.randn(2, 7) * .1)

    def forward(self, input_ids, use_cache=False):
        hidden = self.embedding(input_ids).cumsum(dim=1)
        return SimpleNamespace(logits=self.head(hidden) + hidden @ self.lora_A @ self.lora_B)


def make_turn(model, prompt, completion):
    turn = {'prompt_ids': prompt, 'completion_ids': completion, 'raw_text': '{}', 'finish_reason': 'stop',
            'policy_tensor_sha256': tensor_fingerprint(lora_parameters(model), torch)}
    with torch.no_grad():
        values = token_logps(model, turn, torch, device='cpu').tolist()
    return {**turn, 'old_logps': list(values), 'teacher_logps': list(values)}


def batch(model):
    return [[{'sample_id': f'scene-{g}', 'reward': reward,
              'turns': [make_turn(model, [1, 2 + g], [3 + (m % 2)])]
              + ([make_turn(model, [1, 2 + g, 3, 4], [5, 6])] if m % 2 else [])}
             for m, reward in enumerate([-3., -1., 2., 5.])] for g in range(2)]


def optimizer(model):
    return torch.optim.AdamW(lora_parameters(model).values(), lr=PILOT['learning_rate'], weight_decay=0.)


def test_frozen_configuration():
    config = json.loads((ROOT / 'configs/research/grpo_step30_pilot_v1.json').read_text())
    assert validate_config(config) == PILOT
    for key, value in [('top_p', .95), ('beta', .04), ('planned_steps', 50), ('formal_training_authorized', True), ('seed', True)]:
        changed = {**config, key: value}
        with pytest.raises(ValueError):
            validate_config(changed)


def test_group_normalization_sample_std_and_zero_signal():
    actual = group_advantages([0., 1., 2., 3.])
    expected = (torch.arange(4, dtype=torch.float64) - 1.5) / (torch.arange(4, dtype=torch.float64).std() + 1e-4)
    assert actual == pytest.approx(expected.tolist())
    assert sum(actual) == pytest.approx(0.)
    assert group_advantages([3.] * 4) == [0.] * 4
    with pytest.raises(ValueError):
        group_advantages([1., math.nan])


@pytest.mark.parametrize(('ratio', 'advantage', 'expected'), [(1.5, 1., -1.2), (.5, 1., -.5), (1.5, -1., 1.5), (.5, -1., .8)])
def test_signed_clipped_surrogate(ratio, advantage, expected):
    value = clipped_loss(torch.tensor([math.log(ratio)]), torch.tensor([0.]), advantage, torch)
    assert value.item() == pytest.approx(expected)


def test_shifted_generated_positions_only():
    model = TinyPolicy()
    turn = make_turn(model, [1, 2, 3], [4, 5])
    full = model(torch.tensor([[1, 2, 3, 4, 5]])).logits
    expected = -torch.nn.functional.cross_entropy(full[0, 2:4].float(), torch.tensor([4, 5]), reduction='none')
    actual = token_logps(model, turn, torch, device='cpu')
    assert actual.shape == (2,)
    # Equivalent FP32 logsumexp vs fused CE kernels need not be bit-identical
    # across ARM/x86. Still reject a real one-token shift or wrong target IDs.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    shifted = -torch.nn.functional.cross_entropy(full[0, 1:3].float(), torch.tensor([4, 5]), reduction='none')
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual, shifted, rtol=1e-6, atol=1e-6)


def test_real_adamw_matches_single_graph_trajectory_weighting():
    model = TinyPolicy()
    reference = copy.deepcopy(model)
    groups = batch(model)
    native_optimizer, reference_optimizer = optimizer(model), optimizer(reference)
    loss = torch.tensor(0.)
    for group in groups:
        for member, advantage in zip(group, group_advantages([r['reward'] for r in group]), strict=True):
            length = sum(len(t['completion_ids']) for t in member['turns'])
            for turn in member['turns']:
                new = token_logps(reference, turn, torch, device='cpu')
                loss = loss + clipped_loss(new, torch.tensor(turn['old_logps']), advantage, torch).sum() / (length * 8)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_parameters(reference).values(), PILOT['max_grad_norm'])
    reference_optimizer.step()
    result = optimizer_update(model, native_optimizer, groups, PILOT, torch, device='cpu')
    assert result['effective'] and result['optimizer_steps'] == 1 and result['changed_tensors'] == 2
    assert result['max_same_policy_logprob_delta'] == 0 and result['grad_norm'] > 0
    assert result['loss'] == pytest.approx(loss.item(), abs=1e-6)
    for name, value in model.named_parameters():
        assert torch.allclose(value, dict(reference.named_parameters())[name], atol=2e-8, rtol=1e-7)
        if 'lora_' not in name:
            assert value.grad is None


def test_zero_variance_never_calls_optimizer_step():
    model = TinyPolicy()
    groups = batch(model)
    for group in groups:
        for member in group:
            member['reward'] = 5.
    before = copy.deepcopy(model.state_dict())
    opt = optimizer(model)
    result = optimizer_update(model, opt, groups, PILOT, torch, device='cpu')
    assert not result['effective'] and result['optimizer_steps'] == 0 and not opt.state
    assert all(torch.equal(before[n], p) for n, p in model.state_dict().items())


@pytest.mark.parametrize('corruption', ['incomplete', 'mixed_scene', 'stale_policy', 'teacher_drift', 'nan', 'wrong_lr'])
def test_bad_batch_stops_before_update(corruption):
    model = TinyPolicy()
    groups = batch(model)
    opt = optimizer(model)
    if corruption == 'incomplete':
        groups[0].pop()
    elif corruption == 'mixed_scene':
        groups[0][0]['sample_id'] = 'other-scene'
    elif corruption == 'stale_policy':
        groups[0][0]['turns'][0]['policy_tensor_sha256'] = '0' * 64
    elif corruption == 'teacher_drift':
        groups[0][0]['turns'][0]['teacher_logps'][0] += .1
    elif corruption == 'nan':
        groups[0][0]['turns'][0]['old_logps'][0] = math.nan
    else:
        opt.param_groups[0]['lr'] = 1e-3
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        optimizer_update(model, opt, groups, PILOT, torch, device='cpu')
    assert not opt.state and all(torch.equal(before[n], p) for n, p in model.state_dict().items())


def test_actual_sampling_denominator_not_overwritten_by_teacher():
    model = TinyPolicy()
    groups = batch(model)
    for group in groups:
        for member in group:
            for turn in member['turns']:
                turn['old_logps'] = [v + .1 for v in turn['old_logps']]
    result = optimizer_update(model, optimizer(model), groups, PILOT, torch, device='cpu')
    assert result['max_same_policy_logprob_delta'] == 0
    assert result['max_sampling_teacher_logprob_delta'] == pytest.approx(.1, abs=1e-6)
    assert result['initial_ratio_min'] == pytest.approx(math.exp(-.1), abs=1e-6)
    assert result['initial_ratio_max'] == pytest.approx(math.exp(-.1), abs=1e-6)


@pytest.fixture
def train_scene():
    records, _ = load_registered_development(ROOT, split='train')
    return next(r for r in records if r['reference_depth'] == 1)


def returned(text, finish='stop'):
    return {'prompt_ids': [1, 2], 'completion_ids': [3], 'old_logps': [-1.],
            'raw_text': text, 'finish_reason': finish}


@pytest.mark.parametrize('version', ['reward-v1-legacy', 'reward-v2-research-v1'])
def test_native_multiturn_uses_unchanged_trace_reward_not_final_answer(train_scene, version):
    plan, _ = protocol_plan(train_scene)
    responses = [returned('<tool_call>' + json.dumps({'name': plan[0]['name'], 'arguments': plan[0]['arguments']}) + '</tool_call>'),
                 returned('THIS IS NOT A CORRECT FINAL ANSWER')]
    seen = []
    def generate(messages, seed, limit):
        seen.append(copy.deepcopy(messages))
        return responses[len(seen) - 1]
    result = rollout(train_scene, PILOT, version, 123, generate=generate)
    assert len(result['turns']) == 2 and result['termination'] == 'finished'
    assert seen[1][-1]['role'] == 'tool'
    assert result['reward_breakdown'] == score(train_scene, result['trace'], version)


@pytest.mark.parametrize('text,finish,termination', [('<tool_call>broken', 'stop', 'model_protocol_error'), ('{}', 'length', 'truncated')])
def test_model_errors_keep_actions(train_scene, text, finish, termination):
    result = rollout(train_scene, PILOT, 'reward-v2-research-v1', 9,
                     generate=lambda *args: returned(text, finish))
    assert result['termination'] == termination and len(result['turns']) == 1
    assert result['turns'][0]['completion_ids'] == [3] and result['reward'] <= 2


def test_no_test_access_no_hidden_infrastructure_retry(train_scene):
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError('synthetic GPU failure')
    with pytest.raises(ValueError, match='train-only'):
        rollout({**train_scene, 'split': 'test'}, PILOT, 'reward-v1-legacy', 9, generate=fail)
    assert not calls
    with pytest.raises(RuntimeError):
        rollout(train_scene, PILOT, 'reward-v1-legacy', 9, generate=fail)
    assert calls == [1]
    def overbudget(*args):
        raise PromptBudgetExceeded('before first action')
    with pytest.raises(ValueError, match='no sampled actions'):
        rollout(train_scene, PILOT, 'reward-v1-legacy', 9, generate=overbudget)


def test_gpu_flags_required_before_identity_or_forward():
    with pytest.raises(SystemExit) as error:
        entry.main(['--base-dir', '/unused', '--sft-run-dir', '/unused', '--output', '/unused', '--run-gpu'])
    assert error.value.code == 2


def test_select_step30_binding_and_reject_old_step(tmp_path, monkeypatch):
    selection = {'best': {'step': 30, 'identity': {'fingerprint': entry.SELECTED_FP}},
                 'best_model_spec': {'checkpoint_step': 30, 'kind': 'checkpoint'}}
    (tmp_path / 'selection.json').write_text(json.dumps(selection))
    expected = {'fingerprint': entry.SELECTED_FP, 'base_fingerprint': entry.BASE_FP}
    monkeypatch.setattr(entry, 'resolve_identity', lambda spec: expected)
    _, identity = entry.selected_source(tmp_path, tmp_path)
    assert identity == expected
    selection['best']['step'] = 10
    (tmp_path / 'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='selected SFT30'):
        entry.selected_source(tmp_path, tmp_path)


def test_default_cli_never_starts_gpu(tmp_path, monkeypatch):
    plan = {'state': 'CPU-preflight-no-model-load'}
    monkeypatch.setattr(entry, 'prepare', lambda args: (plan, None, None, None, None))
    def forbidden(*args, **kwargs):
        raise AssertionError('default preflight must not launch workers')
    monkeypatch.setattr(entry.subprocess, 'run', forbidden)
    output = tmp_path / 'new-preflight'
    entry.main(['--base-dir', str(tmp_path / 'base'), '--sft-run-dir', str(tmp_path / 'sft'), '--output', str(output)])
    assert json.loads((output / 'plan.json').read_text()) == plan


def test_real_hf_sampling_scores_and_teacher_snapshot_cpu_only():
    # Legacy ART CPU tests install fake SDK modules in sys.modules. Mirror the
    # independent production worker: don't import real HF through those stubs.
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_assert_hf_sampling_cpu']()"
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve())],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1'),
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mode', ['pass', 'source-drift', 'no-effective'])
def test_parent_binds_two_fresh_workers_and_rejects_drift(tmp_path, monkeypatch, mode):
    plan = {'state': 'CPU-preflight-no-model-load', 'identity': {'fingerprint': entry.SELECTED_FP},
            'config': PILOT, 'schedule': [['same-train-IDs-and-seeds']],
            'train_source': {'sha256': 'train'}, 'validation_source': {'sha256': 'validation'},
            'source_hashes': {'native-script': 'sha'}}
    monkeypatch.setattr(entry, 'prepare', lambda args: (plan, None, None, None, None))
    calls = []
    def worker(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 1800 and '--confirm-two-arm-pilot' in command
        output = Path(command[command.index('--output') + 1]); output.mkdir()
        arm = command[command.index('--arm') + 1]
        receipt = {'state': 'synthetic-CPU-test-worker', 'effective_updates': 0 if mode == 'no-effective' else 1,
                   'initial_adapter_tensor_sha256': 'same-step30-tensors', 'fresh_reload_verified': mode != 'no-effective',
                   'source_hashes': plan['source_hashes']}
        worker_plan = copy.deepcopy(plan)
        if mode == 'source-drift':
            worker_plan['source_hashes'] = {'native-script': 'changed'}
        (output/'plan.json').write_text(json.dumps(worker_plan))
        (output/'result.json').write_text(json.dumps(receipt))
        assert arm == ('G1' if len(calls) == 1 else 'G2')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess, 'run', worker)
    argv = ['--base-dir',str(tmp_path/'base'),'--sft-run-dir',str(tmp_path/'sft'),
            '--output',str(tmp_path/'run'),'--run-gpu','--confirm-two-arm-pilot']
    if mode == 'source-drift':
        with pytest.raises(ValueError, match='differs from parent'):
            entry.main(argv)
        assert len(calls) == 1
    elif mode == 'no-effective':
        with pytest.raises(SystemExit) as error:
            entry.main(argv)
        assert error.value.code == 2 and len(calls) == 2
    else:
        entry.main(argv)
        assert len(calls) == 2
        assert json.loads((tmp_path/'run/result.json').read_text())['state'] == 'pilot-passed-not-formal-results'


def _assert_hf_sampling_cpu():
    # Randomly initialized MINIATURE model, no pretrained weights/network/GPU.
    import peft
    import transformers
    config = transformers.Qwen2Config(vocab_size=7, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=6, pad_token_id=6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        base_model = transformers.Qwen2ForCausalLM(config)
        base_state = copy.deepcopy(base_model.state_dict())
        model = peft.get_peft_model(base_model,
            peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0., target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    class MiniTokenizer:
        eos_token_id = 6
        def apply_chat_template(self, *args, **kwargs):
            return transformers.BatchEncoding({'input_ids': torch.tensor([[1, 2]]),
                                                'attention_mask': torch.tensor([[1, 1]])})
        def decode(self, ids, **kwargs):
            return ' '.join(map(str, ids))
    events = []
    sampler = entry.Sampler(model, MiniTokenizer(), torch, transformers, PILOT, events.append, device='cpu')
    a, b = sampler([], 37, 3), sampler([], 37, 3)
    assert a == b and 0 < len(a['completion_ids']) <= 3
    assert len(a['old_logps']) == len(a['teacher_logps']) == len(a['completion_ids'])
    assert a['old_logps'] == pytest.approx(a['teacher_logps'], abs=1e-5)
    assert sampler.requests == 2 and sampler.completion_tokens == 2 * len(a['completion_ids'])
    assert [e['kind'] for e in events] == ['request-start', 'request-returned'] * 2
    # CPU synthetic GROUPS validate the native optimizer plus real PEFT
    # save/fresh-base reload and AdamW state serialization. Not capability data.
    import tempfile
    opt = optimizer(model)
    receipt = optimizer_update(model, opt, batch(model), PILOT, torch, device='cpu')
    assert receipt['effective']
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model.save_pretrained(str(root/'adapter'), safe_serialization=True, save_embedding_layers=False)
        fresh_base = transformers.Qwen2ForCausalLM(config)
        fresh_base.load_state_dict(base_state)
        reloaded = peft.PeftModel.from_pretrained(fresh_base, root/'adapter', is_trainable=True, local_files_only=True)
        expected, actual = peft.get_peft_model_state_dict(model), peft.get_peft_model_state_dict(reloaded)
        assert set(expected) == set(actual) and all(torch.equal(v, actual[k]) for k, v in expected.items())
        model.eval(); reloaded.eval()
        ids = torch.tensor([[1, 2, 3]])
        assert torch.equal(model(input_ids=ids).logits, reloaded(input_ids=ids).logits)
        contract = {'version': PILOT['version'], 'arm': 'synthetic-CPU-only'}
        saved = save_resume(root/'state', model, opt, torch, step=1, contract=contract, cuda=False)
        restored_optimizer = optimizer(reloaded)
        restored = restore_resume(root/'state', reloaded, restored_optimizer, torch,
                                  expected_contract=contract, device='cpu')
        assert json.loads(json.dumps(saved)) == restored and restored['next_step'] == 2
        optimizer_update(model, opt, batch(model), PILOT, torch, device='cpu')
        optimizer_update(reloaded, restored_optimizer, batch(reloaded), PILOT, torch, device='cpu')
        assert all(torch.equal(v, dict(reloaded.named_parameters())[k]) for k, v in model.named_parameters())

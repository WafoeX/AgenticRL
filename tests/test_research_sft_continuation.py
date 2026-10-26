"""Continuation/control checks; CPU fixtures are not a real SFT run."""
import argparse
import asyncio
import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import scripts.continue_research_sft as continuation
from src.research_sft_run import batch_indices, restore_resume, save_resume


@pytest.fixture(autouse=True)
def isolated_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def initial():
    config = json.loads((continuation.ROOT / 'configs/research/sft_short_v1.json').read_text())
    return {'step': 10, 'config': config}


def candidate(step, score, dynamic=0):
    return {'step': step, 'answer_correct': score, 'dynamic_valid': dynamic,
            'comparison_valid': True, 'scenarios': 180, 'model_spec': {'checkpoint_path': f'/fixture/{step}'},
            'identity': {'step': step}}


def test_total30_means20_additional_updates_and_two_validations():
    configs = continuation.segment_configs(initial(), 30, [20, 30])
    assert [c['terminal_step'] for c in configs] == [20, 30]
    assert [c['candidate_steps'] for c in configs] == [[20], [30]]
    original = initial()['config']
    assert all(c['learning_rate'] == original['learning_rate'] for c in configs)
    indices = [i for s in range(1, 31) for i in batch_indices(1200, s, original)]
    assert len(indices) == len(set(indices)) == 360
    assert not set(indices[:120]) & set(indices[120:])


@pytest.mark.parametrize('target,points', [(10, [10]), (30, [20]), (30, [30, 20]), (30, [20, 20, 30])])
def test_invalid_extension_rejected(target, points):
    with pytest.raises(ValueError):
        continuation.segment_configs(initial(), target, points)


def test_can_resume30_to50_without_changing_training_contract():
    parent = initial()
    parent['config'] = continuation.segment_configs(parent, 30, [20, 30])[-1]
    parent['step'] = 30
    assert [c['terminal_step'] for c in continuation.segment_configs(parent, 50, [40, 50])] == [40, 50]


def test_best_retains_old_model_and_ties_choose_earlier(monkeypatch, tmp_path):
    def fake_identity(spec):
        return {'fingerprint': 'same-model'}
    monkeypatch.setattr(continuation, 'resolve_identity', fake_identity)
    source = tmp_path / 'source'
    source.mkdir()
    for name in ['adapter_config.json', 'research_checkpoint.json']:
        (source / name).write_text('{}')
    (source / 'adapter_model.safetensors').write_bytes(b'CPU-fixture-not-model-weights')
    (source / 'continuation-state').mkdir()
    (source / 'continuation-state/optimizer.txt').write_text('not copied into inference best')
    old = {**candidate(10, 74, 48), 'model_spec': {'checkpoint_path': str(source)}, 'identity': fake_identity({})}
    worse = candidate(20, 60, 40)
    tied = candidate(30, 74, 48)
    result = continuation.publish_best(tmp_path / 'run', [old, worse, tied])
    assert result['best']['step'] == 10
    assert (Path(result['best_model_path']) / 'adapter_model.safetensors').is_file()
    assert not (Path(result['best_model_path']) / 'continuation-state').exists()
    again = continuation.publish_best(tmp_path / 'run', [old, worse, tied])
    assert again['best_model_path'] == result['best_model_path']


def test_best_can_be20_while_last30_remains_resumable(tmp_path, monkeypatch):
    parent = {**initial(), 'checkpoint_path': '/fixture/10', 'identity': {'step': 10},
              'contract': {'source_hashes': {}}}
    parents = {'/fixture/10': parent}
    monkeypatch.setattr(continuation.core, 'load_resume_source', lambda p: parents[str(p)])
    monkeypatch.setattr(continuation, 'check_parent', lambda *_: {})
    monkeypatch.setattr(continuation, 'load_lineage', lambda _: [candidate(5, 2), candidate(10, 74)])
    published = []
    def publish(output, candidates):
        best = continuation.select_checkpoint(candidates, sorted(c['step'] for c in candidates))
        published.append(best['step'])
        return {'best': best, 'best_model_path': str(output / 'best' / str(best['step'])), 'candidates': candidates}
    monkeypatch.setattr(continuation, 'publish_best', publish)
    calls = []
    async def execute(args, folder, config, source, result):
        end = config['terminal_step']
        calls.append((source['step'], end))
        ckpt = str(folder / f'checkpoint-{end}')
        row = candidate(end, 90 if end == 20 else 80)
        row.update(model_spec={'checkpoint_path': ckpt}, identity={'step': end})
        continuation.write_json(folder / 'selection.json', {'candidates': [row]})
        result.update(actual_step=end, actual_updates_this_run=end-source['step'], last_checkpoint=ckpt,
                      last_committed_step=end, training_performed=True)
        parents[ckpt] = {'step': end, 'config': copy.deepcopy(config), 'checkpoint_path': ckpt,
                         'identity': {'step': end}, 'contract': {'source_hashes': {}}}
    args = argparse.Namespace(resume_from=Path('/fixture/10'), base_dir=tmp_path/'base',
                              output=tmp_path/'run', target_step=30, eval_steps=[20, 30], run_gpu=True)
    result = asyncio.run(continuation.run(args, executor=execute))
    assert calls == [(10, 20), (20, 30)] and published == [10, 20, 20]
    assert result['actual_step'] == 30 and result['best_step'] == 20
    assert result['next_training_step'] == 31
    assert result['resume_from_last'].endswith('stage-30/checkpoint-30')
    assert (tmp_path/'run/stage-30/lineage-selection.json').is_file()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_vector = torch.nn.Parameter(torch.tensor([0.1, -0.2]))


def test_actual_CPU_AdamW_and_RNG_resume10_20_30_equals_uninterrupted30(tmp_path):
    contract = {'fixture': 'CPU-only-not-official-model'}
    def setup():
        model = Tiny()
        return model, torch.optim.AdamW(model.parameters(), lr=1e-4)
    def steps(model, optimizer, first, last):
        for _ in range(first, last + 1):
            optimizer.zero_grad()
            (model.lora_vector - torch.randn_like(model.lora_vector)).square().sum().backward()
            optimizer.step()
    torch.manual_seed(73)
    full, full_opt = setup()
    steps(full, full_opt, 1, 30)
    expected = full.lora_vector.detach().clone()
    torch.manual_seed(73)
    model, optimizer = setup()
    for start, end in [(1, 10), (11, 20), (21, 30)]:
        steps(model, optimizer, start, end)
        directory = tmp_path / str(end)
        save_resume(directory, model, optimizer, torch, step=end, contract=contract, cuda=False)
        save_file(model.state_dict(), str(directory/'adapter.safetensors'))
        model, optimizer = setup()
        model.load_state_dict(load_file(str(directory/'adapter.safetensors')))
        restored = restore_resume(directory, model, optimizer, torch, expected_contract=contract, device='cpu')
        assert restored['next_step'] == end + 1
    assert torch.equal(model.lora_vector.detach(), expected)
    assert all(s['step'].item() == 30 for s in optimizer.state.values())


def test_json_receipt_preserves_runtime_objects_and_matches_saved_keys():
    receipt = {'corpus': {'depth_counts': {1: 360, 2: 480, 3: 360}}, 'tokenized_rows_sha256': 'same'}
    runtime = [object(), object(), object(), object()]
    normalized, *returned = continuation.json_prepared_metadata((receipt, *runtime))
    assert normalized == json.loads(json.dumps(receipt))
    assert set(normalized['corpus']['depth_counts']) == {'1', '2', '3'}
    assert set(receipt['corpus']['depth_counts']) == {1, 2, 3}
    assert all(a is b for a, b in zip(runtime, returned, strict=True))
    assert continuation.metadata_differences(receipt, normalized) == []


@pytest.mark.parametrize('fail', [False, True])
def test_actual_execute_boundary_normalizes_metadata_and_restores_binding(monkeypatch, fail):
    receipt = {'corpus': {'depth_counts': {1: 360}}}
    sentinel = object()
    def prepare(*_):
        return (receipt, sentinel, sentinel, sentinel, sentinel)
    monkeypatch.setattr(continuation.core, 'prepare', prepare)
    async def execute(*_):
        observed = continuation.core.prepare(Path('/CPU-fixture'), {})
        assert observed[0]['corpus']['depth_counts'] == {'1': 360}
        assert observed[1] is sentinel
        if fail:
            raise ValueError('CPU test interruption')
    monkeypatch.setattr(continuation.core, 'execute', execute)
    if fail:
        with pytest.raises(ValueError, match='CPU test interruption'):
            asyncio.run(continuation.execute_with_json_metadata(None, None, None, None, None))
    else:
        asyncio.run(continuation.execute_with_json_metadata(None, None, None, None, None))
    assert continuation.core.prepare is prepare


def test_changed_token_hash_still_rejected_after_normalization(monkeypatch, tmp_path):
    config = initial()['config']
    receipt = {'corpus': {'depth_counts': {1: 360}}, 'tokenized_rows_sha256': 'original'}
    expected_receipt = json.loads(json.dumps(receipt))
    contract = continuation.resume_contract(config, expected_receipt, {'torch': 'fixture'})
    contract['source_hashes'] = {}
    parent = {'contract': contract}
    monkeypatch.setattr(continuation.importlib.metadata, 'version', lambda _: 'fixture')
    monkeypatch.setattr(continuation.core, 'prepare', lambda *_: (receipt, None, None, None, None))
    assert continuation.check_parent(tmp_path, parent, config) == expected_receipt
    receipt['tokenized_rows_sha256'] = 'changed'
    with pytest.raises(ValueError, match='tokenized_rows_sha256'):
        continuation.check_parent(tmp_path, parent, config)


def test_serialized_receipt_contract_restores_real_CPU_AdamW_state(tmp_path):
    config = initial()['config']
    receipt = {'corpus': {'depth_counts': {1: 360, 2: 480, 3: 360}}}
    saved_contract = json.loads(json.dumps(continuation.resume_contract(config, receipt, {'fixture': 'CPU'})))
    normalized = continuation.json_prepared_metadata((receipt, None, None, None, None))[0]
    current_contract = continuation.resume_contract(config, normalized, {'fixture': 'CPU'})
    assert saved_contract == current_contract
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(10):
        optimizer.zero_grad()
        model.lora_vector.square().sum().backward()
        optimizer.step()
    save_resume(tmp_path/'state', model, optimizer, torch, step=10, contract=saved_contract, cuda=False)
    restored_model = Tiny()
    restored_model.load_state_dict(model.state_dict())
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    restored = restore_resume(tmp_path/'state', restored_model, restored_optimizer, torch,
                              expected_contract=current_contract, device='cpu')
    assert restored['next_step'] == 11
    assert all(s['step'].item() == 10 for s in restored_optimizer.state.values())

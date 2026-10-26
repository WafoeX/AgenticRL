"""Continue committed SFT checkpoints; retain lineage best and resumable last.

Uses the unchanged SFT execute/save/restore functions. One segment per validation
point makes best publication complete before the next segment starts. Default
step10 -> 20 -> 30; no GPU/model forward without --run-gpu.
"""
import argparse
import asyncio
import copy
import importlib.metadata
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_research_sft as core
from src.research_model_identity import artifact_inventory, file_hash, resolve_identity
from src.research_schema import load_json
from src.research_sft_run import (
    resume_contract,
    select_checkpoint,
    validate_config,
    write_json,
)

PLAN = ROOT / 'configs/research/sft_to30_v2.json'


def segment_configs(parent, target, points):
    start = parent['step']
    if type(target) is not int or target <= start or not points or points != sorted(set(points)):
        raise ValueError('target must advance and validation points must be distinct/in order')
    if any(type(p) is not int or p <= 0 or p > target for p in points) or points[-1] != target:
        raise ValueError('validation must include the terminal step')
    steps = [p for p in points if p > start]
    if not steps:
        raise ValueError('no future validation points')
    result, previous = [], parent['config']
    for end in steps:
        config = copy.deepcopy(previous)
        config.update(version=f'research-sft-short-continued-{end}-v2', terminal_step=end, candidate_steps=[end])
        validate_config(config, parent=previous, start_step=start)
        result.append(config)
        start, previous = end, config
    return result


def load_lineage(checkpoint):
    folder = Path(checkpoint).parent
    path = folder / 'lineage-selection.json'
    saved = load_json(path if path.exists() else folder / 'selection.json')
    candidates = saved['candidates']
    select_checkpoint(candidates, sorted({c['step'] for c in candidates}))
    return candidates


def publish_best(output, candidates):
    """Keep each published best model; never delete or overwrite prior models."""
    best = select_checkpoint(candidates, sorted({c['step'] for c in candidates}))
    spec, identity = best['model_spec'], best['identity']
    if resolve_identity(spec) != identity:
        raise ValueError('selected source checkpoint changed')
    source = Path(spec['checkpoint_path'])
    destination = output / 'best' / f"checkpoint-{best['step']}"
    files = artifact_inventory(source)
    if destination.exists():
        if artifact_inventory(destination) != files:
            raise ValueError('existing best copy differs')
    else:
        destination.mkdir(parents=True)
        for name in files:
            shutil.copy2(source / name, destination / name)
    copied_spec = {**spec, 'checkpoint_path': str(destination.resolve())}
    copied_identity = resolve_identity(copied_spec)
    if copied_identity['fingerprint'] != identity['fingerprint']:
        raise ValueError('best copy differs from selected model')
    selection = {'best': best, 'best_model_path': str(destination.resolve()), 'best_model_spec': copied_spec,
                 'candidates': candidates, 'scope': 'evaluated-lineage-checkpoints-only', 'test_used': False}
    write_json(output / 'selection.json', selection)
    print(f"BEST step={best['step']} answer={best['answer_correct']}/180 dynamic={best['dynamic_valid']} path={destination}", flush=True)
    return selection


def persisted_json_metadata(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('continuation metadata must have the persisted JSON representation') from exc


def json_prepared_metadata(prepared):
    """Use the exact JSON representation already used by checkpoint receipts.

    build_corpus.depth_counts has integer keys in memory; JSON saves string
    keys. Only receipt metadata is normalized. Training rows, tokenizer,
    validation records and protocol objects are returned untouched.
    """
    receipt, *runtime = prepared
    serialized = persisted_json_metadata(receipt)
    return (serialized, *runtime)


def metadata_differences(saved, current, path='contract', limit=12):
    differences = []
    def visit(a, b, location):
        if len(differences) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(location + '.' + key + ': missing field')
                else:
                    visit(a[key], b[key], location + '.' + key)
                if len(differences) >= limit:
                    break
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(f'{location}: length {len(a)} -> {len(b)}')
            else:
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    visit(x, y, f'{location}[{i}]')
                    if len(differences) >= limit:
                        break
        elif a != b:
            differences.append(f'{location}: saved={repr(a)[:120]} current={repr(b)[:120]}')
    # Both sides are compared as persisted JSON, never a mixture of int/str keys.
    visit(persisted_json_metadata(saved), persisted_json_metadata(current), path)
    return differences


async def execute_with_json_metadata(args, folder, config, parent, result):
    """Adapt the native metadata boundary for BOTH check and actual restore.

    The native training/save/restore code stays byte-identical. This is not a
    comparison bypass: all data/token hashes and optimizer contracts still match
    exactly. Always restore the module binding, including on failure.
    """
    original_prepare = core.prepare
    def prepare(base, settings):
        return json_prepared_metadata(original_prepare(base, settings))
    core.prepare = prepare
    try:
        await core.execute(args, folder, config, parent, result)
    finally:
        core.prepare = original_prepare


def check_parent(base, parent, config):
    """Restore actual optimizer/RNG contract, not merely adapter warm-start."""
    source_hashes = parent['contract']['source_hashes']
    if any(file_hash(ROOT / name) != sha for name, sha in source_hashes.items()):
        raise ValueError('original SFT core source differs; use the continuation bundle')
    versions = {name: importlib.metadata.version(name) for name in parent['contract']['versions']}
    receipt, _, _, _, _ = json_prepared_metadata(core.prepare(base, config))
    contract = resume_contract(config, receipt, versions)
    contract['source_hashes'] = dict(source_hashes)
    if contract != parent['contract']:
        different = metadata_differences(parent['contract'], contract)
        raise ValueError('resume contract differs after JSON normalization:\n' + '\n'.join(different))
    return receipt


async def run(args, *, executor=execute_with_json_metadata):
    parent = core.load_resume_source(args.resume_from)
    plan = load_json(PLAN)
    target = args.target_step if args.target_step is not None else plan['target_step']
    points = args.eval_steps if args.eval_steps is not None else plan['validation_steps']
    configs = segment_configs(parent, target, points)
    base, output = args.base_dir.resolve(), args.output.resolve()
    source_run = Path(parent['checkpoint_path']).parent
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(base) or output.is_relative_to(source_run):
        raise ValueError('choose a new external Drive run directory, not the source run')
    check_parent(base, parent, configs[0])
    candidates = load_lineage(parent['checkpoint_path'])
    if not any(c['step'] == parent['step'] and c['identity'] == parent['identity'] for c in candidates):
        raise ValueError('parent checkpoint is missing from validated lineage')
    output.mkdir(parents=True, exist_ok=False)
    result = {'version': 'research-sft-continuation-v2.1-json-metadata', 'state': 'preflight-only', 'start_step': parent['step'],
              'target_step': target, 'additional_planned_updates': target - parent['step'],
              'validation_steps': [c['terminal_step'] for c in configs], 'new_validation_tasks': 180 * len(configs),
              'parent_checkpoint': parent['checkpoint_path'], 'training_performed': False, 'test_used': False,
              'GRPO_training_started': False, 'core_source_hashes': parent['contract']['source_hashes'],
              'orchestrator_sha256': file_hash(Path(__file__).resolve()),
              'metadata_boundary': 'persisted-JSON-receipt-normalization-v2',
              'started_at': datetime.now(UTC).isoformat(), 'total_GPU_budget_verified': False}
    write_json(output / 'plan.json', {**result, 'segment_configs': configs})
    write_json(output / 'result.json', result)
    if not args.run_gpu:
        return result
    selection = publish_best(output, candidates)  # Retain old best even if later training/validation fails.
    result['state'] = 'running'
    begun = time.monotonic()
    try:
        for config in configs:
            end = config['terminal_step']
            folder = output / f'stage-{end}'
            folder.mkdir(exist_ok=False)
            segment = {'version': 'research-sft-run-v1', 'state': 'preparing', 'config': config,
                       'monotonic_start': time.monotonic(), 'actual_updates_this_run': 0, 'actual_step': parent['step'],
                       'last_committed_step': None, 'last_checkpoint': None, 'training_performed': False,
                       'test_used': False, 'GRPO_training_started': False,
                       'source_hashes': dict(parent['contract']['source_hashes']),
                       'orchestrator_sha256': result['orchestrator_sha256'],
                       'source_provenance': 'unchanged-core-by-hash; JSON-receipt-boundary-adapter; standalone-orchestrator-not-a-new-git-commit'}
            write_json(folder / 'result.json', segment)
            native_args = argparse.Namespace(base_dir=base, run_gpu=True)
            print(f"CONTINUE {parent['step']} -> {end}; validation180; checkpoint+optimizer/RNG saved on Drive", flush=True)
            try:
                await executor(native_args, folder, config, parent, segment)
            except BaseException as exc:
                segment.update(state='failed', error_type=type(exc).__name__)
                raise
            finally:
                write_json(folder / 'result.json', segment)
                result['training_performed'] = result['training_performed'] or segment['training_performed']
                result['actual_step'] = segment['actual_step']
                if segment['last_checkpoint'] is not None:
                    result.update(last_checkpoint=segment['last_checkpoint'],
                                  last_committed_step=segment['last_committed_step'])
            if segment['actual_step'] != end or segment['actual_updates_this_run'] != end - parent['step']:
                raise ValueError('segment did not complete the declared updates')
            current = load_json(folder / 'selection.json')['candidates']
            select_checkpoint(current, [end])
            candidates = [*candidates, *current]
            selection = publish_best(output, candidates)
            write_json(folder / 'lineage-selection.json', selection)
            last_checkpoint = segment['last_checkpoint']
            if not isinstance(last_checkpoint, str) or segment['last_committed_step'] != end:
                raise ValueError('terminal checkpoint/continuation state was not committed')
            parent = core.load_resume_source(Path(last_checkpoint))
            result.update(training_performed=True, actual_step=end, last_checkpoint=parent['checkpoint_path'],
                          best_step=selection['best']['step'], best_model_path=selection['best_model_path'])
            write_json(output / 'result.json', result)
        result.update(state='completed', resume_from_last=parent['checkpoint_path'],
                      resume_marker=str(Path(parent['checkpoint_path']).parent / f"continuation-{target}.json"),
                      next_training_step=target + 1)
    except BaseException as exc:
        result.update(state='failed', error_type=type(exc).__name__)
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - begun
        write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-step', type=int)
    parser.add_argument('--eval-steps', type=int, nargs='+')
    parser.add_argument('--run-gpu', action='store_true')
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

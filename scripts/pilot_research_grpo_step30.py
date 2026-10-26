#!/usr/bin/env python3
"""NEW step30 native-GRPO pilot. Default CPU preflight; no ART/final test.

Explicit --run-gpu --confirm-two-arm-pilot launches two SEPARATE processes.
No formal-training flags, no retry, no automatic extension or baseline rerun.
"""
import argparse
import asyncio
import copy
import gc
import importlib
import json
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_research_baselines import REVISION, VERSIONS
from scripts.pilot_research_sft import load_base
from scripts.research_grpo_native import (
    PILOT,
    optimizer_update,
    rollout,
    token_logps,
    validate_config,
)
from scripts.train_research_sft import append_json, spec_for, validate_candidate
from src.research_environment import TOOL_SCHEMAS
from src.research_eval_backend import PromptBudgetExceeded
from src.research_grpo import member_seed, scene_plan
from src.research_grpo_runtime import verify_fresh_optimizer
from src.research_model_identity import file_hash, resolve_identity, verify_unchanged
from src.research_registration import load_registered_development
from src.research_schema import load_json, model_messages
from src.research_sft_pilot import tensor_fingerprint
from src.research_sft_run import lora_parameters, save_resume, write_json

SELECTED_FP = '467e2a3253af826c71aa0f96cadcf495e59705b7af4bd3c8aa4466e91ad31633'
BASE_FP = '5572b0e15dbd0cc2e6cc4f23aac1bab454184c1ad2170cd5bcd64230e535864c'


def selected_source(base, run):
    selection = load_json(run / 'selection.json')
    if selection['best']['step'] != 30 or selection['best']['identity']['fingerprint'] != SELECTED_FP:
        raise ValueError('requires reviewed selected SFT30, not step10/20 or a different run')
    spec = dict(selection['best_model_spec'])
    if spec['checkpoint_step'] != 30 or spec['kind'] != 'checkpoint':
        raise ValueError('selected best spec must be checkpoint30')
    spec['base_path'] = str(base.resolve(strict=True))
    identity = resolve_identity(spec)
    if identity['fingerprint'] != SELECTED_FP or identity['base_fingerprint'] != BASE_FP:
        raise ValueError('selected SFT30/Base bytes differ from reviewed feedback')
    return spec, identity


def source_hashes():
    files = list((ROOT / 'src').glob('*.py')) + [Path(__file__).resolve(), ROOT / 'scripts/train_research_sft.py',
        ROOT / 'scripts/pilot_research_sft.py', ROOT / 'scripts/eval_research_baselines.py',
        ROOT / 'scripts/research_grpo_native.py',
        ROOT / 'configs/research/grpo_step30_pilot_v1.json', ROOT / 'requirements-sft-pilot.txt']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def prepare(args):
    config = validate_config(load_json(ROOT / 'configs/research/grpo_step30_pilot_v1.json'))
    spec, identity = selected_source(args.base_dir, args.sft_run_dir)
    training, source = load_registered_development(ROOT, split='train')
    validation, val_source = load_registered_development(ROOT, split='validation')
    if len(training) != 1200 or len(validation) != 180:
        raise ValueError('fixed train/validation cardinality differs')
    schedule = scene_plan(training, config)
    tokenizer = importlib.import_module('transformers').AutoTokenizer.from_pretrained(
        identity['base_path'], local_files_only=True, trust_remote_code=False, use_fast=True)
    for batch in schedule:
        for record in batch:
            prefix = tokenizer.apply_chat_template(model_messages(record), tools=TOOL_SCHEMAS,
                                                   tokenize=True, add_generation_prompt=True)
            if len(prefix) > config['max_prompt_tokens']:
                raise ValueError('initial train prompt exceeds pilot budget')
    protocol = load_json(ROOT / 'configs/research/evaluation_v1.json')
    protocol['modes'] = ['deterministic']
    count = 2 * config['planned_steps'] * config['groups_per_step'] * config['group_size']
    plan = {'version': config['version'], 'state': 'CPU-preflight-no-model-load', 'model_spec': spec,
            'identity': identity, 'config': config, 'source_hashes': source_hashes(),
            'train_source': source, 'validation_source': val_source,
            'schedule': [[{'sample_id': r['sample_id'], 'depth': r['reference_depth'],
                          'seeds': [member_seed(config, step, group, m) for m in range(config['group_size'])]}
                         for group, r in enumerate(batch)] for step, batch in enumerate(schedule)],
            'two_arm_max_rollouts': count, 'two_arm_max_sampling_requests': count * config['max_turns'],
            'two_arm_max_completion_tokens': count * config['max_total_completion_tokens'],
            'two_arm_max_prompt_tokens': count * config['max_turns'] * config['max_prompt_tokens'],
            'conditional_validation_tasks': 360, 'formal_training_authorized': False,
            'historical_remaining_GPU_budget_verified': False, 'test_used': False}
    return plan, tokenizer, schedule, validation, protocol


class Sampler:
    def __init__(self, model, tokenizer, torch, transformers, config, log, *, device='cuda:0'):
        self.model, self.tokenizer, self.torch, self.transformers = model, tokenizer, torch, transformers
        self.device = device
        self.config, self.log, self.context = config, log, {}
        self.policy_fp = tensor_fingerprint(lora_parameters(model), torch)
        self.requests = self.prompt_tokens = self.completion_tokens = 0

    def __call__(self, messages, seed, limit):
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(messages, tools=copy.deepcopy(TOOL_SCHEMAS), tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.device)
        prefix = inputs['input_ids'][0].tolist()
        if len(prefix) > self.config['max_prompt_tokens']:
            raise PromptBudgetExceeded('native prompt budget')
        self.requests += 1
        self.log({'kind': 'request-start', 'request': self.requests, **self.context,
                  'seed': seed, 'prompt_tokens': len(prefix), 'max_new_tokens': limit})
        generation = self.transformers.GenerationConfig(max_new_tokens=limit, do_sample=True,
            temperature=1.0, top_p=1.0, top_k=0, num_beams=1, num_return_sequences=1,
            return_dict_in_generate=True, output_scores=True, max_time=120,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        self.model.eval()
        with torch.random.fork_rng(devices=[0] if self.device.startswith('cuda') else []), torch.inference_mode():
            torch.manual_seed(seed)
            output = self.model.generate(**inputs, generation_config=generation, use_model_defaults=False)
            ids = output.sequences[0, len(prefix):].tolist()
            if len(output.scores) != len(ids) or not ids:
                raise ValueError('missing sampled scores/token IDs; no fallback LP allowed')
            sampled = [torch.log_softmax(s[0].float(), dim=-1)[i].item()
                       for s, i in zip(output.scores, ids, strict=True)]
        # Log real generated cost before the optional teacher-forced audit forward.
        self.prompt_tokens += len(prefix)
        self.completion_tokens += len(ids)
        self.log({'kind': 'request-returned', 'request': self.requests, **self.context,
                  'prompt_tokens': len(prefix), 'completion_tokens': len(ids)})
        del output, inputs
        stopped = ids[-1] == self.tokenizer.eos_token_id
        turn = {'prompt_ids': prefix, 'completion_ids': ids, 'old_logps': sampled,
                'policy_tensor_sha256': self.policy_fp,
                'raw_text': self.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False),
                'finish_reason': 'stop' if stopped else 'length'}
        with torch.no_grad():
            turn['teacher_logps'] = token_logps(self.model, turn, torch, device=self.device).cpu().tolist()
        return turn


async def run_worker(args, output):
    plan, tokenizer, schedule, validation, protocol = prepare(args)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    config, identity = plan['config'], plan['identity']
    result = {'state': 'loading', 'arm': args.arm, 'reward_version': config['rewards'][args.arm],
              'initial_identity': identity, 'planned_steps': 2, 'effective_updates': 0,
              'source_hashes': plan['source_hashes'], 'formal_training': False, 'test_used': False,
              'training_performed': False, 'fresh_reload_verified': False}
    started, model, torch, sampler = time.monotonic(), None, None, None
    write_json(output / 'result.json', result)
    log = lambda row: append_json(output / 'events.jsonl', row)
    try:
        versions = {p: metadata.version(p) for p in VERSIONS}
        if any(versions[p].split('+')[0] != v for p, v in VERSIONS.items()):
            raise ValueError('restore fixed SFT environment, not ART/Unsloth')
        torch = importlib.import_module('torch')
        transformers = importlib.import_module('transformers')
        peft = importlib.import_module('peft')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError('exactly one visible BF16 CUDA GPU required')
        props = torch.cuda.get_device_properties(0)
        if 'A100' not in props.name or props.total_memory < 39 * 1024**3:
            raise ValueError('pilot requires A100 >=40GB')
        torch.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        result.update(GPU=props.name, versions=versions, device='cuda:0')
        model = peft.PeftModel.from_pretrained(load_base(identity, torch, transformers), identity['checkpoint_path'],
            is_trainable=True, local_files_only=True, use_safetensors=True)
        saved = importlib.import_module('safetensors.torch').load_file(
            str(Path(identity['checkpoint_path']) / 'adapter_model.safetensors'), device='cpu')
        actual = peft.get_peft_model_state_dict(model)
        if set(actual) != set(saved) or any(not torch.equal(actual[k].detach().cpu(), saved[k]) for k in saved):
            raise ValueError('actual loaded adapter not identical to SFT30')
        del actual, saved
        parameters = lora_parameters(model)
        if (len(parameters) != 336 or sum(p.numel() for p in parameters.values()) != 4399104
                or list(model.active_adapters) != ['default']):
            raise ValueError('actual trainable LoRA/default-adapter shape differs from reviewed SFT30')
        optimizer = torch.optim.AdamW(parameters.values(), lr=config['learning_rate'], weight_decay=0.0)
        verify_fresh_optimizer(optimizer, list(parameters.values()))
        result.update(initial_adapter_tensor_sha256=tensor_fingerprint(parameters, torch),
                      fresh_optimizer_state_entries=0, initial_trainable_parameters=sum(p.numel() for p in parameters.values()))
        sampler = Sampler(model, tokenizer, torch, transformers, config, log)
        result['state'] = 'sampling-and-training'
        write_json(output / 'result.json', result)
        for step, batch in enumerate(schedule):
            groups = []
            for group_index, record in enumerate(batch):
                members = []
                for member in range(config['group_size']):
                    sampler.context = {'planned_step': step + 1, 'group': group_index, 'member': member,
                                       'sample_id': record['sample_id']}
                    value = rollout(record, config, config['rewards'][args.arm],
                        member_seed(config, step, group_index, member), generate=sampler)
                    append_json(output / 'rollouts.jsonl', {**sampler.context, 'payload': value})
                    members.append(value)
                groups.append(members)
            receipt = optimizer_update(model, optimizer, groups, config, torch, device='cuda:0')
            result['effective_updates'] += receipt['optimizer_steps']
            result['training_performed'] = result['effective_updates'] > 0
            log({'kind': 'update', 'planned_step': step + 1, **receipt})
            sampler.policy_fp = tensor_fingerprint(parameters, torch)
            print(f"{args.arm}: planned={step + 1}/2 effective={result['effective_updates']} {receipt}", flush=True)
            write_json(output / 'result.json', result)
        count = result['effective_updates']
        result['state'] = 'saving-and-evaluating'
        write_json(output / 'result.json', result)
        checkpoint = output / f'checkpoint-{count}'
        model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
        write_json(checkpoint / 'research_checkpoint.json',
                   {'step': count, 'base_fingerprint': identity['base_fingerprint'], 'base_revision': REVISION})
        spec = spec_for(args.base_dir, REVISION, checkpoint, count)
        final_identity = resolve_identity(spec)
        result.update(checkpoint=str(checkpoint), final_identity=final_identity,
                      final_adapter_tensor_sha256=tensor_fingerprint(parameters, torch))
        if count:
            save_resume(output / f'optimizer-state-{count}', model, optimizer, torch, step=count,
                        contract={'version': config['version'], 'arm': args.arm, 'config': config,
                                  'parent_fingerprint': SELECTED_FP}, cuda=True)
            row = await validate_candidate(output / f'validation-{count}', spec, final_identity,
                model, tokenizer, validation, protocol, versions, torch, {'validation_scenarios': 180})
            result.update(validation=row, fresh_reload_verified=True, state='pilot-completed-not-effectiveness-claim')
        else:
            result['state'] = 'no-effective-update-not-accepted'
        verify_unchanged(plan['model_spec'], identity)
        if source_hashes() != plan['source_hashes']:
            raise ValueError('execution source changed during worker')
    except BaseException as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if sampler is not None:
            result['sampling_cost'] = {'requests': sampler.requests, 'prompt_tokens_returned': sampler.prompt_tokens,
                'completion_tokens_returned': sampler.completion_tokens,
                'pending_request_usage_may_be_unknown': result['state'] == 'failed'}
        result['wall_seconds'] = time.monotonic() - started
        if torch is not None and torch.cuda.is_available():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write_json(output / 'result.json', result)
        sampler = model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--sft-run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--confirm-two-arm-pilot', action='store_true')
    parser.add_argument('--arm', choices=['G1', 'G2'], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_gpu != args.confirm_two_arm_pilot or args.arm and not args.run_gpu:
        parser.error('GPU requires BOTH explicit pilot flags; no formal training flag exists')
    output = args.output.expanduser().resolve()
    for source in [args.sft_run_dir.resolve(), args.base_dir.resolve()]:
        if output.is_relative_to(source) or source.is_relative_to(output):
            parser.error('pilot outputs must not overlap SFT/Base source')
    if args.arm:
        asyncio.run(run_worker(args, output))
        return
    plan, _, _, _, _ = prepare(args)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    if not args.run_gpu:
        print(json.dumps({'state': plan['state'], 'selected_step': 30, 'pilot_rollouts': 32,
                          'model_forward_performed': False}))
        return
    summary = {'state': 'running', 'arms': {}, 'formal_training': False, 'test_used': False}
    write_json(output / 'result.json', summary)
    try:
        for arm in ['G1', 'G2']:
            command = [sys.executable, str(Path(__file__).resolve()), '--base-dir', str(args.base_dir.resolve()),
                '--sft-run-dir', str(args.sft_run_dir.resolve()), '--output', str(output / arm),
                '--run-gpu', '--confirm-two-arm-pilot', '--arm', arm]
            print(f"Starting {arm}; live progress is written to {output / f'{arm}.log'}; cap 1800 seconds", flush=True)
            with (output / f'{arm}.log').open('x') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=PILOT['arm_wall_seconds'], check=False)
            if completed.returncode:
                raise RuntimeError(f'{arm} worker failed: inspect {arm}.log; do not rerun training blindly')
            summary['arms'][arm] = load_json(output / arm / 'result.json')
            worker_plan = load_json(output / arm / 'plan.json')
            if (any(worker_plan[k] != plan[k] for k in
                    ['identity', 'config', 'schedule', 'train_source', 'validation_source', 'source_hashes'])
                    or summary['arms'][arm]['source_hashes'] != plan['source_hashes']):
                raise ValueError('worker source/data/config/schedule differs from parent preflight')
            print(f"Finished {arm}: {summary['arms'][arm]['state']}", flush=True)
            write_json(output / 'result.json', summary)
        a, b = (summary['arms'][arm] for arm in ['G1', 'G2'])
        if a['initial_adapter_tensor_sha256'] != b['initial_adapter_tensor_sha256']:
            raise ValueError('G1/G2 initial loaded tensor fingerprints differ')
        accepted = all(r['effective_updates'] > 0 and r['fresh_reload_verified'] for r in (a, b))
        summary.update(state='pilot-passed-not-formal-results' if accepted else 'no-signal-pilot-not-accepted',
                       actual_train_save_reload_verified=accepted)
    except BaseException as exc:
        summary.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'result.json', summary)
    print(json.dumps({'state': summary['state'], 'arms_effective_updates':
                      {a: r['effective_updates'] for a, r in summary['arms'].items()}}))
    if not accepted:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

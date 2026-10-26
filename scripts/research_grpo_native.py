"""Native on-policy, multi-turn GRPO; heavy dependencies injected/lazy.

A NEW execution backend, not an ART compatibility workaround/acceptance receipt.
All assistant tokens (including EOS/malformed/truncated responses) are actions.
Prompt/system/user/tool tokens are conditioning only. No answer-oracle reward.
"""
import copy
import math
import statistics

from src.research_environment import ResearchEnvironment
from src.research_eval_backend import (
    ModelProtocolError,
    PromptBudgetExceeded,
    decode_qwen,
)
from src.research_grpo_reward import score
from src.research_schema import model_messages
from src.research_sft_pilot import tensor_fingerprint
from src.research_sft_run import lora_parameters

PILOT = {
    "version": "step30-native-grpo-pilot-v1", "seed": 20261011,
    "planned_steps": 2, "groups_per_step": 2, "group_size": 4,
    "learning_rate": 5e-6, "beta": 0.0, "clip_epsilon": 0.2,
    "advantage_epsilon": 1e-4, "std_correction": 1, "optimizer_epochs": 1,
    "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "max_completion_tokens": 512, "max_prompt_tokens": 3584,
    "max_total_completion_tokens": 1536, "max_turns": 6,
    "max_grad_norm": 1.0, "old_logprob_atol": 1e-4,
    "rewards": {"G1": "reward-v1-legacy", "G2": "reward-v2-research-v1"},
    "automatic_retries": 0, "validation_scenarios": 180,
    "arm_wall_seconds": 1800, "formal_training_authorized": False,
}


def validate_config(config):
    if config != PILOT or any(type(config[k]) is not type(v) for k, v in PILOT.items()):
        raise ValueError("fixed step30 two-arm pilot only; formal budgets need a new version")
    return copy.deepcopy(config)


def group_advantages(rewards, *, epsilon=1e-4):
    if len(rewards) < 2 or any(type(r) not in (int, float) or not math.isfinite(r) for r in rewards):
        raise ValueError("complete finite reward group required")
    deviation = statistics.stdev(rewards)  # sample std, correction=1, as TRL GRPO
    mean = statistics.mean(rewards)
    return [(r - mean) / (deviation + epsilon) for r in rewards]


def clipped_loss(new_logps, old_logps, advantage, torch, *, epsilon=0.2):
    ratio = torch.exp(new_logps - old_logps)
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)


def token_logps(model, turn, torch, *, device):
    """Exact per-request prefix; never retokenize reconstructed concatenated chats."""
    prompt, completion = turn['prompt_ids'], turn['completion_ids']
    if not prompt or not completion or any(type(i) is not int or i < 0 for i in prompt + completion):
        raise ValueError("nonempty exact prompt/action token IDs required")
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    output = model(input_ids=ids, use_cache=False)
    # token P is predicted by logit P-1; only generated assistant positions train.
    logits = output.logits[0, len(prompt) - 1:-1].float()
    target = ids[0, len(prompt):]
    return logits.gather(-1, target[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def validate_turn(turn, config, limit):
    if (not turn['prompt_ids'] or not turn['completion_ids']
            or len(turn['prompt_ids']) > config['max_prompt_tokens']
            or len(turn['completion_ids']) > limit
            or len(turn['old_logps']) != len(turn['completion_ids'])
            or any(type(i) is not int or i < 0 for i in turn['prompt_ids'] + turn['completion_ids'])
            or any(not math.isfinite(v) or v > 1e-5 for v in turn['old_logps'])
            or turn['finish_reason'] not in ('stop', 'length')):
        raise ValueError("sampled token/logprob/budget contract differs")


def rollout(record, config, reward_version, seed, *, generate):
    """generate(messages, seed, limit) returns exact IDs/raw text/old logps.

    Infrastructure exceptions abort the pilot (no hidden retry/negative reward).
    Model protocol mistakes remain real, rewarded trajectories with their IDs.
    """
    if record['split'] != 'train':
        raise ValueError("train-only rollout")
    result = {'sample_id': record['sample_id'], 'seed': seed, 'messages': model_messages(record),
              'turns': [], 'trace': [], 'termination': 'truncated'}
    environment = ResearchEnvironment(record['environment_snapshot'])
    for index in range(config['max_turns']):
        used = sum(len(t['completion_ids']) for t in result['turns'])
        remaining = config['max_total_completion_tokens'] - used
        if remaining <= 0:
            break
        limit = min(config['max_completion_tokens'], remaining)
        try:
            turn = generate(copy.deepcopy(result['messages']), (seed + index) % (2**31), limit)
        except PromptBudgetExceeded:
            result['termination'] = 'prompt_budget'
            break
        validate_turn(turn, config, limit)
        result['turns'].append(turn)
        try:
            message = decode_qwen(turn['raw_text'], index)
        except ModelProtocolError:
            result['messages'].append({'role': 'assistant', 'content': turn['raw_text']})
            result['termination'] = 'model_protocol_error'
            break
        result['messages'].append(message)
        if turn['finish_reason'] == 'length':
            break  # retain sampled tokens; never execute a truncated tool call
        calls = message.get('tool_calls', [])
        if not calls:
            result['termination'] = 'finished'
            break
        for call in calls:
            function = call['function']
            observation = environment.execute(function['name'], function['arguments'])
            result['trace'].append({'turn': index, 'name': function['name'],
                                    'arguments': function['arguments'], 'observation': observation})
            result['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'content': observation})
    breakdown = score(record, result['trace'], reward_version)
    # Same truncation/protocol completion cap as the historical research rollout.
    if result['termination'] != 'finished' and 'completion' in breakdown:
        raw = sum(v for k, v in breakdown.items() if k not in {'version', 'total', 'cap', 'completion'})
        breakdown['completion'], breakdown['cap'] = 0.0, min(0.0, 2.0 - raw)
        breakdown['total'] = raw + breakdown['cap']
    result.update(reward=breakdown['total'], reward_breakdown=breakdown)
    if not result['turns']:
        raise ValueError("no sampled actions; stop, do not silently drop a member")
    return result


def optimizer_counts(optimizer):
    try:
        values = [state['step'].item() for state in optimizer.state.values()]
    except (KeyError, AttributeError, RuntimeError) as exc:
        raise ValueError('invalid actual AdamW step counters') from exc
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or v != math.trunc(v) for v in values):
        raise ValueError('nonfinite/noninteger actual optimizer step counters')
    return values


def optimizer_update(model, optimizer, groups, config, torch, *, device):
    """One optimizer step over ALL complete groups; one epoch, beta=0.

    Objective = mean_trajectory(mean_generated_token(-min(r*A,clip(r)*A))).
    A is group-standardized; all turns of a trajectory share A. Zero-advantage
    members remain in the outer denominator. The denominator uses ACTUAL sampled
    logprobs; a distinct teacher-forced snapshot checks no policy drift. Cached
    generation vs full-prefix numerics may differ; never replace sampled LPs.
    Loss may be zero with positive gradients (not an SFT cross-entropy loss).
    """
    validate_config(config)
    if (len(groups) != config['groups_per_step']
            or any(len(g) != config['group_size'] for g in groups)):
        raise ValueError("complete declared group batch required")
    if (any(len({m['sample_id'] for m in g}) != 1 for g in groups)
            or len({g[0]['sample_id'] for g in groups}) != len(groups)):
        raise ValueError("one unique train scene per complete group required")
    parameters = lora_parameters(model)
    policy_fingerprint = tensor_fingerprint(parameters, torch)
    owned = [p for g in optimizer.param_groups for p in g['params']]
    if (len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters.values()}
            or any(g['lr'] != config['learning_rate'] or g['weight_decay'] != 0 for g in optimizer.param_groups)):
        raise ValueError("optimizer ownership/LR/weight decay differs")
    advantages = [group_advantages([r['reward'] for r in group]) for group in groups]
    if not any(a != 0 for group in advantages for a in group):
        return {'effective': False, 'optimizer_steps': 0, 'reason': 'no_reward_variance', 'changed_tensors': 0}
    before = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    counts_before = optimizer_counts(optimizer)
    model.eval()  # eval is NOT no_grad: exact same dropout-free policy as sampling
    optimizer.zero_grad(set_to_none=True)
    total_loss, max_delta, action_count, sampling_delta = 0.0, 0.0, 0, 0.0
    ratio_min, ratio_max, clipped_count = math.inf, -math.inf, 0
    denominator = sum(len(group) for group in groups)
    try:
        for group, values in zip(groups, advantages, strict=True):
            for member, advantage in zip(group, values, strict=True):
                size = sum(len(t['completion_ids']) for t in member['turns'])
                if not 0 < size <= config['max_total_completion_tokens'] or not 0 < len(member['turns']) <= config['max_turns']:
                    raise ValueError("empty/overbudget policy member")
                action_count += size
                for turn in member['turns']:
                    validate_turn(turn, config, config['max_completion_tokens'])
                    if turn['policy_tensor_sha256'] != policy_fingerprint:
                        raise ValueError("rollouts belong to a different/stale policy")
                    with torch.set_grad_enabled(advantage != 0):
                        new = token_logps(model, turn, torch, device=device)
                    old = torch.tensor(turn['old_logps'], dtype=new.dtype, device=device)
                    teacher = torch.tensor(turn['teacher_logps'], dtype=new.dtype, device=device)
                    if teacher.shape != new.shape or not torch.isfinite(old).all().item():
                        raise ValueError("teacher/sample logprob alignment differs")
                    delta = (new.detach() - teacher).abs().max().item()
                    max_delta = max(max_delta, delta)
                    sampling_delta = max(sampling_delta, (new.detach() - old).abs().max().item())
                    ratios = torch.exp(new.detach() - old)
                    if not torch.isfinite(ratios).all().item():
                        raise ValueError("nonfinite importance ratios")
                    ratio_min, ratio_max = min(ratio_min, ratios.min().item()), max(ratio_max, ratios.max().item())
                    clipped_count += ((ratios - 1).abs() > config['clip_epsilon']).sum().item()
                    if not torch.isfinite(new).all().item() or not math.isfinite(delta) or delta > config['old_logprob_atol']:
                        raise ValueError("same-policy logprob drift; stop before optimizer.step")
                    if advantage == 0:
                        del new, old, teacher, ratios
                        continue
                    loss = clipped_loss(new, old, advantage, torch, epsilon=config['clip_epsilon']).sum() / (size * denominator)
                    if not torch.isfinite(loss).item():
                        raise ValueError("nonfinite GRPO loss")
                    total_loss += loss.item()
                    loss.backward()
                    del new, old, loss, teacher, ratios
        grads = [p.grad for p in parameters.values() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all().item() for g in grads):
            raise ValueError("missing/nonfinite GRPO gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters.values(), config['max_grad_norm']).item()
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("no effective GRPO gradient")
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    changed = sum(not torch.equal(before[n], p.detach().cpu()) for n, p in parameters.items())
    counts_after = optimizer_counts(optimizer)
    if not changed or max(counts_after) - max(counts_before, default=0) != 1:
        raise ValueError("optimizer did not make exactly one effective update")
    return {'effective': True, 'optimizer_steps': 1, 'changed_tensors': changed,
            'loss': total_loss, 'grad_norm': norm, 'learning_rate': config['learning_rate'], 'beta': 0.0,
            'max_same_policy_logprob_delta': max_delta, 'sampled_policy_tokens': action_count,
            'max_sampling_teacher_logprob_delta': sampling_delta,
            'initial_ratio_min': ratio_min, 'initial_ratio_max': ratio_max,
            'initial_clip_fraction': clipped_count / action_count,
            'trajectory_denominator': denominator, 'advantages': advantages,
            'adapter_tensor_sha256': tensor_fingerprint(parameters, torch)}

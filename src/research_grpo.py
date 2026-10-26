"""CPU-testable GRPO control plane; lazy ART integration, no GPU launch on import.

Transport adapters must translate ONLY known service/network failures to InfraFailure.
Programming errors propagate. Model mistakes remain scored, including malformed calls.
"""
import asyncio
import copy
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .research_answers import reject_constant, unique_object
from .research_environment import TOOL_SCHEMAS, ResearchEnvironment
from .research_grpo_reward import score
from .research_schema import digest, model_messages


class InfraFailure(RuntimeError):
    pass


@dataclass
class TrainingRollout:
    messages_and_choices: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    generations: list = field(default_factory=list)
    reward: float | None = None
    reward_breakdown: dict = field(default_factory=dict)
    termination: str = "finished"
    infra_error: str | None = None
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_unknown_requests: int = 0

    def audit(self):
        return {"messages_and_choices": [x if isinstance(x, dict) else x.model_dump(mode="json")
                                         for x in self.messages_and_choices],
                **{k: v for k, v in vars(self).items() if k != "messages_and_choices"}}


def validate_config(config: dict) -> dict:
    from .research_schema import load_json
    template = load_json(Path(__file__).resolve().parents[1] / "configs/research/grpo_contract_v1.json")
    if set(config) != set(template):
        raise ValueError("missing/unknown GRPO configuration keys")
    c = copy.deepcopy(config)
    if (c["version"] != template["version"] or c["status"] != template["status"]
            or type(c["formal_training_authorized"]) is not bool or c["formal_training_authorized"]
            or type(c["final_test_ready"]) is not bool or c["final_test_ready"]
            or c["rewards"] != template["rewards"] or c["whole_group_retries"] != 1
            or c["retry_policy"] != template["retry_policy"]):
        raise ValueError("development-only version/arm/retry contract mismatch")
    for key in ("seed", "planned_steps", "groups_per_step", "group_size", "max_completion_tokens",
                "max_prompt_tokens", "max_total_completion_tokens", "max_turns"):
        if type(c[key]) is not int or c[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid " + key)
    if c["group_size"] < 2 or c["seed"] != 20261011:
        raise ValueError("fixed registered seed and at least two members required")
    for key in ("learning_rate", "beta", "temperature", "top_p", "timeout_seconds"):
        if type(c[key]) not in (int, float) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError("invalid " + key)
    if c["learning_rate"] <= 0 or c["temperature"] <= 0 or not 0 < c["top_p"] <= 1 or c["timeout_seconds"] <= 0:
        raise ValueError("invalid sampling/optimizer configuration")
    if c["max_prompt_tokens"] + c["max_completion_tokens"] > 4096 or c["lora"] != template["lora"]:
        raise ValueError("context/selected-SFT LoRA contract mismatch")
    return c


def member_seed(config, step, group, member):
    # No arm or retry term: arms and full-group retries use identical sampling seeds.
    return int(digest([config["seed"], step, group, member])[:8], 16) % (2**31)


def scene_plan(records, config):
    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("nonempty train-only data required")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate train scene")
    if config["groups_per_step"] > len(records):
        raise ValueError("unique scenes per batch required")
    ids = list(range(len(records)))
    random.Random(config["seed"]).shuffle(ids)
    # Consecutive groups across shuffle wrap; no arm-specific ordering.
    return [[records[ids[(step * config["groups_per_step"] + g) % len(ids)]]
             for g in range(config["groups_per_step"])] for step in range(config["planned_steps"])]


def choice_message(choice):
    message = choice.message.model_dump(mode="json", exclude_none=True)
    message["role"] = "assistant"
    return message


def validate_generation(choice, usage):
    """ART 0.5.17 vLLM serves sampled token IDs as token_id:N strings."""
    if usage is None or any(type(getattr(usage, k, None)) is not int or getattr(usage, k) < 0
                            for k in ("prompt_tokens", "completion_tokens")):
        raise InfraFailure("missing/invalid usage")
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content or len(content) != usage.completion_tokens:
        raise InfraFailure("missing/misaligned policy token logprobs")
    ids, logprobs = [], []
    for token in content:
        match = re.fullmatch(r"token_id:(\d+)", token.token)
        if match is None or not math.isfinite(token.logprob) or token.logprob > 0:
            raise InfraFailure("invalid policy token ID/logprob")
        ids.append(int(match[1]))
        logprobs.append(token.logprob)
    return {"token_ids": ids, "logprobs": logprobs, "assistant_mask": [True] * len(ids)}


async def training_rollout(model_name, record, config, reward_version, seed, *, complete, count_prompt):
    """complete(**OpenAI kwargs) returns raw ChatCompletion, no hidden retries.

    count_prompt uses the actual tokenizer/template + tools, not character counts.
    Assistant Choices remain the original objects. Tool/system/user are dictionaries,
    hence ART excludes their tokens from its training mask.
    """
    if record["split"] != "train":
        raise ValueError("training may not consume validation/test")
    result = TrainingRollout(messages_and_choices=model_messages(record))
    chat = copy.deepcopy(result.messages_and_choices)
    env = ResearchEnvironment(record["environment_snapshot"])
    usage_known = True
    try:
        for turn in range(config["max_turns"]):
            remaining = config["max_total_completion_tokens"] - result.completion_tokens
            if remaining <= 0:
                result.termination = "truncated"
                break
            prompt_count = count_prompt(chat, TOOL_SCHEMAS)
            if prompt_count > config["max_prompt_tokens"]:
                result.termination = "prompt_budget"
                break
            limit = min(config["max_completion_tokens"], remaining)
            result.requests += 1
            usage_known = False
            try:
                response = await asyncio.wait_for(complete(model=model_name, messages=copy.deepcopy(chat),
                    tools=TOOL_SCHEMAS, temperature=config["temperature"], top_p=config["top_p"],
                    max_tokens=limit, seed=(seed + turn) % (2**31), logprobs=True, top_logprobs=1,
                    extra_body={"return_tokens_as_token_ids": True}), timeout=config["timeout_seconds"])
            except TimeoutError as exc:
                raise InfraFailure("request timeout; remote usage unknown") from exc
            usage = response.usage
            if usage is not None and all(type(getattr(usage, k, None)) is int and getattr(usage, k) >= 0
                                         for k in ("prompt_tokens", "completion_tokens")):
                result.prompt_tokens += usage.prompt_tokens
                result.completion_tokens += usage.completion_tokens
                usage_known = True
            if not response.choices or len(response.choices) != 1:
                raise InfraFailure("empty/multiple choices")
            choice = response.choices[0]
            # Keep original Choice even if the training payload is later invalid.
            result.messages_and_choices.append(choice)
            payload = validate_generation(choice, usage)
            if usage is None:
                raise InfraFailure("missing usage")
            if usage.prompt_tokens != prompt_count or usage.completion_tokens > limit:
                raise InfraFailure("server tokenizer/usage/budget mismatch")
            payload["turn"] = turn
            result.generations.append(payload)
            message = choice_message(choice)
            chat.append(message)
            if choice.finish_reason == "length":
                result.termination = "truncated"
            calls = message.get("tool_calls", [])
            if not calls:
                break
            if not isinstance(calls, list):
                result.termination = "model_protocol_error"
                break
            seen = set()
            for call in calls:
                try:
                    fn, call_id = call["function"], call["id"]
                    if not isinstance(call_id, str) or not call_id or call_id in seen:
                        raise ValueError("bad call ID")
                    seen.add(call_id)
                    args = fn["arguments"]
                    args = json.loads(args, object_pairs_hook=unique_object, parse_constant=reject_constant) if isinstance(args, str) else args
                    if not isinstance(args, dict) or not isinstance(fn["name"], str):
                        raise TypeError("bad arguments/name")
                except (KeyError, TypeError, ValueError):
                    result.termination = "model_protocol_error"
                    break
                fn["arguments"] = args  # Native Qwen history, without altering the raw Choice.
                observation = env.execute(fn["name"], args)
                result.trace.append({"turn": turn, "name": fn["name"], "arguments": args, "observation": observation})
                tool = {"role": "tool", "tool_call_id": call_id, "content": observation}
                chat.append(tool)
                result.messages_and_choices.append(tool)
            if result.termination != "finished":
                break
        else:
            result.termination = "truncated"
    except InfraFailure as exc:
        result.infra_error = str(exc)
        result.termination = "infra_failure"
        # No fabricated negative reward. Partial known usage still counts as cost.
        if not usage_known:
            result.usage_unknown_requests += 1
        return result
    result.reward_breakdown = score(record, result.trace, reward_version)
    # Truncation/protocol errors retain the sample, but cannot claim completion.
    if result.termination != "finished" and "completion" in result.reward_breakdown:
        b = result.reward_breakdown
        raw = sum(v for k, v in b.items() if k not in {"version", "total", "cap", "completion"})
        b["completion"], b["cap"] = 0.0, min(0.0, 2.0 - raw)
        b["total"] = raw + b["cap"]
    result.reward = result.reward_breakdown["total"]
    return result


async def collect_group(record, config, step, group_index, *, rollout, log):
    """Persist every member, including successful members of discarded attempts."""
    for attempt in range(config["whole_group_retries"] + 1):
        # No per-member exception filtering, no ART gather dropping exceptions.
        members = []
        for i in range(config["group_size"]):
            member = await rollout(record, member_seed(config, step, group_index, i))
            members.append(member)
            log({"kind": "rollout", "step": step, "group": group_index, "attempt": attempt,
                 "member": i, "sample_id": record["sample_id"],
                 "seed": member_seed(config, step, group_index, i), "payload": member.audit()})
        invalid = any(m.infra_error is not None or m.reward is None for m in members)
        log({"kind": "group", "step": step, "group": group_index, "attempt": attempt,
             "sample_id": record["sample_id"], "size": len(members), "discarded": invalid,
             "variance": None if invalid else statistics.pvariance([m.reward for m in members]),
             "retry": invalid and attempt < config["whole_group_retries"],
             "skipped": invalid and attempt == config["whole_group_retries"]})
        if not invalid:
            if any(not math.isfinite(m.reward) for m in members):
                raise ValueError("nonfinite reward")
            return members
    return None


def art_train_kwargs(config):
    return {"learning_rate": config["learning_rate"], "kl_penalty_coef": config["beta"],
            "allow_training_without_logprobs": False, "scale_rewards": True,
            "scale_learning_rate_by_reward_std_dev": False,
            "num_trajectories_learning_rate_multiplier_power": 0.0, "save_checkpoint": True}


async def run_steps(records, config, *, arm, rollout, update, log):
    """update receives complete raw groups; returns a VERIFIED optimizer receipt.

    Receipt contract is stricter than ART step: optimizer_steps + changed_tensors,
    loss/grad norm, actual lr/beta and saved checkpoint. Production bridge must
    measure these from trainer/tensors; a returned step number is insufficient.
    """
    config = validate_config(config)
    if arm not in config["rewards"]:
        raise ValueError("unknown arm")
    random.seed(config["seed"])  # ART packing uses global Python RNG; both arms reset it.
    begun = time.monotonic()
    totals = {"planned_updates": config["planned_steps"], "backend_calls": 0,
              "effective_updates": 0, "optimizer_steps": 0, "skipped_groups": 0,
              "zero_variance_groups": 0, "retained_rollouts": 0}
    log({"kind": "config", "arm": arm, "config": config, "config_sha256": digest(config)})
    for step, rows in enumerate(scene_plan(records, config)):
        groups = []
        for gi, row in enumerate(rows):
            members = await collect_group(row, config, step, gi, rollout=rollout, log=log)
            if members is None:
                totals["skipped_groups"] += 1
            else:
                groups.append(members)
                totals["retained_rollouts"] += len(members)
                totals["zero_variance_groups"] += statistics.pvariance([m.reward for m in members]) == 0
        signal = any(statistics.pvariance([m.reward for m in group]) > 0 for group in groups)
        if not signal:
            log({"kind": "update", "planned_step": step, "called": False, "reason": "no_reward_variance"})
            continue
        receipt = await update(groups, **art_train_kwargs(config))
        totals["backend_calls"] += 1
        required = {"optimizer_steps", "changed_tensors", "loss", "grad_norm", "learning_rate", "beta", "checkpoint_step", "checkpoint_path"}
        if not required <= receipt.keys():
            raise ValueError("missing optimizer evidence; ART checkpoint step is not an update")
        if (type(receipt["optimizer_steps"]) is not int or receipt["optimizer_steps"] < 0
                or type(receipt["changed_tensors"]) is not int or receipt["changed_tensors"] < 0
                or any(type(receipt[k]) not in (int, float) or not math.isfinite(receipt[k]) for k in ("loss", "grad_norm"))
                or receipt["grad_norm"] < 0 or receipt["learning_rate"] != config["learning_rate"] or receipt["beta"] != config["beta"]):
            raise ValueError("invalid/nonfinite/effective optimizer config mismatch")
        effective = receipt["optimizer_steps"] > 0 and receipt["changed_tensors"] > 0 and receipt["grad_norm"] > 0
        totals["effective_updates"] += effective
        totals["optimizer_steps"] += receipt["optimizer_steps"]
        log({"kind": "update", "planned_step": step, "called": True, "effective": effective, "receipt": receipt})
    totals.update(program_wall_seconds=time.monotonic() - begun, GPU_allocation_hours=None,
                  GPU_allocation_cost_verified=False, actual_ART_GPU_verified=False)
    log({"kind": "summary", **totals})
    return totals


def to_art_groups(groups, art):
    """Only call after collect_group; do not use ART exception-filtering gather."""
    return [art.TrajectoryGroup([art.Trajectory(messages_and_choices=m.messages_and_choices,
            tools=TOOL_SCHEMAS, reward=m.reward, metadata={"termination": m.termination}) for m in group])
            for group in groups]

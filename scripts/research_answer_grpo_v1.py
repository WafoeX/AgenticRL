"""Versioned answer-dominant GRPO policy. No model loading on import.

Reuses the unchanged native sampler/action loop and single-batch optimizer math;
NEVER invokes the old two-arm pilot launcher or changes its global contract.
"""

import copy
import math
import random

from scripts.research_grpo_native import PILOT, optimizer_update, rollout
from src.research_checker import check_messages
from src.research_grpo_reward import score as trace_score

REWARD_VERSION = "reward-answer-dominant-v1"
POLICY_KEYS = [
    "learning_rate",
    "beta",
    "clip_epsilon",
    "advantage_epsilon",
    "std_correction",
    "optimizer_epochs",
    "temperature",
    "top_p",
    "top_k",
    "max_completion_tokens",
    "max_prompt_tokens",
    "max_total_completion_tokens",
    "max_turns",
    "max_grad_norm",
    "old_logprob_atol",
]
CONFIG = {
    "version": "answer-grpo-three-seed-v1",
    "reward_version": REWARD_VERSION,
    "seeds": [20261011, 20261012, 20261013],
    "data_seed": 20261018,
    "planned_batches": 60,
    "groups_per_batch": 2,
    "group_size": 4,
    "validation_batches": [20, 40, 60],
    "save_every_batches": 5,
    "seed_wall_seconds": 9000,
    "campaign_wall_seconds": 27000,
    "validation_reserve_seconds": 1800,
    "answer_weight": 1.0,
    "process_weight": 0.2,
    "policy": {k: PILOT[k] for k in POLICY_KEYS},
    "automatic_retries": 0,
    "final_test_authorized": False,
}


def same_typed(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_typed(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            same_typed(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def validate_config(config):
    if not same_typed(config, CONFIG):
        raise ValueError(
            "fixed typed answer-GRPO v1 plan required; changes need a new version"
        )
    return copy.deepcopy(config)


def answer_reward(record, messages, trace, *, termination):
    """Train-only: finished valid value+unit => 1; process bonus <= .2.

    Thus ANY correct answer outranks ANY wrong answer, even a perfect trace.
    The existing checker and its numerical/unit/format tolerances are unchanged.
    """
    if record.get("split") != "train":
        raise ValueError("answer reward rejects validation/test records")
    if not isinstance(termination, str):
        raise TypeError("termination must be a string")
    if termination not in {
        "finished",
        "truncated",
        "prompt_budget",
        "model_protocol_error",
    }:
        raise ValueError("unknown/infra termination must not receive fabricated reward")
    checker_term = {
        "prompt_budget": "truncated",
        "model_protocol_error": "exception",
    }.get(termination, termination)
    checked = check_messages(record, messages, trace, termination=checker_term)[
        "answer"
    ]
    parts = trace_score(record, trace, "reward-v2-research-v1")
    if termination != "finished" and "completion" in parts:
        raw = sum(
            v
            for k, v in parts.items()
            if k not in {"version", "total", "cap", "completion"}
        )
        parts["completion"], parts["cap"] = 0.0, min(0.0, 2.0 - raw)
        parts["total"] = raw + parts["cap"]
    raw_flow = parts["total"]
    if not math.isfinite(raw_flow) or not -3.000001 <= raw_flow <= 4.000001:
        raise ValueError("legacy V2 shaping outside documented bounds")
    process = max(0.0, min(1.0, (raw_flow + 3.0) / 7.0))
    primary = 1.0 if checked["correct"] else 0.0
    bonus = 0.2 * process
    return {
        "version": REWARD_VERSION,
        "total": primary + bonus,
        "answer_correct": checked["correct"],
        "answer_reason": checked["reason"],
        "answer_component": primary,
        "process_score": process,
        "process_bonus": bonus,
        "flow_v2_raw": raw_flow,
        "termination": termination,
        "train_only": True,
    }


def answer_rollout(record, seed, *, generate):
    # Old loop supplies unchanged Qwen codec, tool execution, raw IDs, truncation
    # and infrastructure propagation. Its temporary trace reward is NOT optimized.
    payload = rollout(record, PILOT, "reward-v2-research-v1", seed, generate=generate)
    reward = answer_reward(
        record,
        payload["messages"],
        payload["trace"],
        termination=payload["termination"],
    )
    payload["reward"], payload["reward_breakdown"] = reward["total"], reward
    return payload


def update_batch(model, optimizer, groups, config, torch, *, device):
    validate_config(config)
    # PILOT is only the immutable single-batch mathematical kernel contract here.
    # Seed, campaign length and validation scheduling are controlled by NEW code;
    # none of those old launcher fields is consulted by optimizer_update.
    return optimizer_update(model, optimizer, groups, PILOT, torch, device=device)


def training_schedule(records, config):
    validate_config(config)
    if len(records) != 1200 or any(r.get("split") != "train" for r in records):
        raise ValueError("exact registered train1200 required")
    if len({r["sample_id"] for r in records}) != 1200:
        raise ValueError("duplicate train ID")
    pools = {d: [r for r in records if r["reference_depth"] == d] for d in [1, 2, 3]}
    for depth, pool in pools.items():
        random.Random(config["data_seed"] + depth).shuffle(pool)
        if len(pool) < 40:
            raise ValueError("insufficient depth stratum")
    # 120 distinct questions, 40/depth; fixed for ALL seeds, never reward-selected.
    order = [pools[depth][i] for i in range(40) for depth in [1, 2, 3]]
    return [order[i : i + 2] for i in range(0, 120, 2)]


def member_seed(config, seed, planned_batch, group, member):
    from src.research_schema import digest

    validate_config(config)
    if (
        seed not in config["seeds"]
        or not 1 <= planned_batch <= 60
        or group not in [0, 1]
        or member not in range(4)
    ):
        raise ValueError("unknown seed/batch/member")
    return int(
        digest([config["version"], seed, planned_batch, group, member])[:8], 16
    ) % (2**31)


def choose_candidate(candidates):
    if not candidates or any(
        not c["comparison_valid"] or c["scenarios"] != 180 for c in candidates
    ):
        raise ValueError("only complete valid180 candidates may be selected")
    return max(
        candidates,
        key=lambda c: (c["answer_correct"], c["dynamic_valid"], -c["planned_batch"]),
    )

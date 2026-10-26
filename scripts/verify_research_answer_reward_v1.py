"""Train-only OFFLINE answer-reward verification. No model load/forward/GPU.
Reads prior TRAIN rollouts, not validation attempts; rescoring is NOT RL training.
"""

import argparse
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from scripts.research_answer_grpo_v1 import CONFIG, answer_reward, training_schedule
from src.research_registration import load_registered_development
from src.research_sft import build_expert
from src.research_sft_run import write_json

ROOT = Path(__file__).resolve().parents[1]
PILOT_SHA = "f7bee2ea7e680a5ff15fcf714a64e0bc29deb1d7cf703d93a0d1f6920b6bbdbf"


def load_rollouts(feedback=None):
    if feedback is not None:
        if hashlib.sha256(feedback.read_bytes()).hexdigest() != PILOT_SHA:
            raise ValueError("requires the reviewed original pilot feedback")
        try:
            with zipfile.ZipFile(feedback) as archive:
                return [
                    {"arm": arm, **json.loads(line)}
                    for arm in ["G1", "G2"]
                    for line in archive.read(f"run/{arm}/rollouts.jsonl")
                    .decode()
                    .splitlines()
                ]
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid original training-rollout archive") from exc
    path = ROOT / "fixtures/research_answer_reward_v1/pilot_train_rollouts.jsonl"
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError) as exc:
        raise ValueError(
            "provide --feedback or the bundled train-only fixture"
        ) from exc


def verify(feedback=None):
    training, _ = load_registered_development(ROOT, split="train")
    lookup = {r["sample_id"]: r for r in training}
    examples = load_rollouts(feedback)
    if len(examples) != 32:
        raise ValueError("expected all 32 pilot training members")
    expert_correct = 0
    for record in training:
        expert = build_expert(record)
        reward = answer_reward(
            record, expert["messages"], expert["trace"], termination="finished"
        )
        if not reward["answer_correct"] or not 1.0 <= reward["total"] <= 1.200001:
            raise ValueError("gold expert is not rewarded as task success")
        expert_correct += 1
    rescored = []
    for row in examples:
        payload = row["payload"]
        reward = answer_reward(
            lookup[row["sample_id"]],
            payload["messages"],
            payload["trace"],
            termination=payload["termination"],
        )
        if not reward["answer_correct"] and not 0 <= reward["total"] <= 0.200001:
            raise ValueError("incorrect final answer dominates success reward")
        rescored.append(
            {
                **{
                    k: row[k]
                    for k in ["arm", "planned_step", "group", "member", "sample_id"]
                },
                "old_reward": payload["reward"],
                "new_reward": reward,
            }
        )
    groups = []
    for arm in ["G1", "G2"]:
        for step in [1, 2]:
            for group in [0, 1]:
                members = [
                    r
                    for r in rescored
                    if r["arm"] == arm
                    and r["planned_step"] == step
                    and r["group"] == group
                ]
                if len(members) != 4:
                    raise ValueError("incomplete prior group")
                old = [r["old_reward"] for r in members]
                new = [r["new_reward"]["total"] for r in members]
                groups.append(
                    {
                        "arm": arm,
                        "batch": step,
                        "group": group,
                        "old_rewards": old,
                        "new_rewards": new,
                        "old_std": statistics.stdev(old),
                        "new_std": statistics.stdev(new),
                    }
                )
    schedule = training_schedule(training, CONFIG)
    coverage = {
        str(d): sum(r["reference_depth"] == d for batch in schedule for r in batch)
        for d in [1, 2, 3]
    }
    return {
        "status": "offline-reward-verified-not-new-training",
        "reward_version": CONFIG["reward_version"],
        "expert_train_checked": expert_correct,
        "prior_train_members_rescored": len(examples),
        "groups": groups,
        "prior_reward4_wrong_answers_now_below_success": [
            r
            for r in rescored
            if r["old_reward"] == 4 and not r["new_reward"]["answer_correct"]
        ],
        "schedule_batches": len(schedule),
        "unique_train_scenes": len({r["sample_id"] for b in schedule for r in b}),
        "depth_coverage": coverage,
        "all_seeds_same_questions": True,
        "validation_reward_calls": 0,
        "test_reward_calls": 0,
        "new_official_model_forward": False,
        "GPU_training": False,
        "original_feedback_sha256": PILOT_SHA,
        "reward_source_sha256": hashlib.sha256(
            (ROOT / "scripts/research_answer_grpo_v1.py").read_bytes()
        ).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve old verification")
    result = verify(args.feedback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "expert_train_checked",
                    "prior_train_members_rescored",
                    "depth_coverage",
                ]
            }
        )
    )

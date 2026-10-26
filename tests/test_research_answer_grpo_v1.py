"""Answer-reward/3-seed scheduling/budget tests: CPU only, no official weights."""

import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_research_answer_grpo_v1 as entry
from scripts.research_answer_grpo_v1 import (
    CONFIG,
    answer_reward,
    answer_rollout,
    choose_candidate,
    member_seed,
    training_schedule,
    update_batch,
    validate_config,
)
from scripts.research_grpo_native import PILOT, token_logps
from src.research_registration import load_registered_development
from src.research_sft import build_expert
from src.research_sft_pilot import tensor_fingerprint
from src.research_sft_run import lora_parameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records():
    return load_registered_development(ROOT, split="train")[0]


def test_fixed_typed_config_and_no_test_or_retry():
    c = json.loads(
        (ROOT / "configs/research/grpo_answer_three_seed_v1.json").read_text()
    )
    assert validate_config(c) == CONFIG
    bad = copy.deepcopy(c)
    bad["policy"]["beta"] = False
    with pytest.raises(ValueError):
        validate_config(bad)
    for k, v in [
        ("campaign_wall_seconds", 43200),
        ("planned_batches", 61),
        ("final_test_authorized", True),
    ]:
        with pytest.raises(ValueError):
            validate_config({**c, k: v})
    assert (
        c["automatic_retries"] == 0
        and c["campaign_wall_seconds"] == 3 * c["seed_wall_seconds"] == 27000
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_answer_dominates_even_perfect_wrong_trace(records, depth):
    record = next(r for r in records if r["reference_depth"] == depth)
    expert = build_expert(record)
    good = answer_reward(
        record, expert["messages"], expert["trace"], termination="finished"
    )
    bad = copy.deepcopy(expert["messages"])
    bad[-1]["content"] = '{"unit":"WRONG_UNIT","value":0}'
    wrong = answer_reward(record, bad, expert["trace"], termination="finished")
    assert good["answer_correct"] and 1 <= good["total"] <= 1.200001
    assert not wrong["answer_correct"] and 0 <= wrong["total"] <= 0.200001
    assert (
        good["total"] > wrong["total"]
        and good["process_score"] == wrong["process_score"]
    )
    no_tools = answer_reward(record, expert["messages"], [], termination="finished")
    assert no_tools["answer_correct"] and no_tools["total"] >= 1 > wrong["total"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_reward_rejects_holdouts_before_parsing(split):
    with pytest.raises(ValueError, match="rejects"):
        answer_reward({"split": split}, [], [], termination="finished")


@pytest.mark.parametrize(
    "termination", ["truncated", "model_protocol_error", "prompt_budget"]
)
def test_unfinished_gold_answer_cannot_receive_success(records, termination):
    e = build_expert(records[0])
    r = answer_reward(records[0], e["messages"], e["trace"], termination=termination)
    assert (
        not r["answer_correct"]
        and r["answer_component"] == 0
        and r["total"] <= 0.200001
    )


def test_infrastructure_not_negative_model_reward(records):
    with pytest.raises(ValueError):
        answer_reward(records[0], [], [], termination="infra_failure")
    with pytest.raises(TypeError):
        answer_reward(records[0], [], [], termination=None)


def test_depth_balanced_unique_same_data_schedule(records):
    a = training_schedule(records, CONFIG)
    b = training_schedule(records, copy.deepcopy(CONFIG))
    assert a == b and len(a) == 60 and all(len(x) == 2 for x in a)
    assert len({r["sample_id"] for batch in a for r in batch}) == 120
    assert [
        sum(r["reference_depth"] == d for batch in a for r in batch) for d in [1, 2, 3]
    ] == [40] * 3
    assert [r["reference_depth"] for batch in a[:3] for r in batch] == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError):
        training_schedule([{**r, "split": "validation"} for r in records], CONFIG)
    with pytest.raises(ValueError):
        training_schedule(records[:-1], CONFIG)


def test_seeds_distinct_reproducible_and_no_arm_pilot():
    values = [member_seed(CONFIG, s, 1, 0, 0) for s in CONFIG["seeds"]]
    assert len(set(values)) == 3 and values == [
        member_seed(CONFIG, s, 1, 0, 0) for s in CONFIG["seeds"]
    ]
    with pytest.raises(ValueError):
        member_seed(CONFIG, 123, 1, 0, 0)
    assert CONFIG["policy"] == {k: PILOT[k] for k in CONFIG["policy"]}


def test_selection_parent_fallback_tie_and_invalid():
    parent = {
        "planned_batch": 0,
        "answer_correct": 128,
        "dynamic_valid": 75,
        "scenarios": 180,
        "comparison_valid": True,
    }
    weaker = {**parent, "planned_batch": 20, "answer_correct": 120}
    equal = {**parent, "planned_batch": 40}
    assert choose_candidate([parent, weaker, equal]) == parent
    better = {**parent, "planned_batch": 60, "answer_correct": 132}
    assert choose_candidate([parent, better]) == better
    with pytest.raises(ValueError):
        choose_candidate([{**better, "comparison_valid": False}])


def test_default_and_flag_guards_do_not_launch_workers(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        entry.main(
            ["--base-dir", "/x", "--sft-run-dir", "/y", "--output", "/z", "--run-gpu"]
        )
    plan = {"state": "CPU-only-test", "config": CONFIG}
    monkeypatch.setattr(entry, "prepare", lambda args: (plan, None, None, None, None))

    def forbidden(*a, **k):
        raise AssertionError("no GPU worker in preflight")

    monkeypatch.setattr(entry, "campaign", forbidden)
    entry.main(
        [
            "--base-dir",
            str(tmp_path / "base"),
            "--sft-run-dir",
            str(tmp_path / "sft"),
            "--output",
            str(tmp_path / "preflight"),
        ]
    )
    assert json.loads((tmp_path / "preflight/plan.json").read_text()) == plan


def test_real_cpu_subprocess_hard_timeout(tmp_path):
    started = time.monotonic()
    r = entry.limited_process(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        tmp_path / "timeout.log",
        0.2,
    )
    assert (
        r["budget_exhausted"]
        and r["returncode"] != 0
        and time.monotonic() - started < 5
    )
    done = entry.limited_process(
        [sys.executable, "-c", 'print("CPU only")'], tmp_path / "done.log", 5
    )
    assert done["returncode"] == 0 and not done["budget_exhausted"]
    assert "CPU only" in (tmp_path / "done.log").read_text()


def test_timeout_kills_own_child_group_not_just_parent(tmp_path):
    log = tmp_path / "tree.log"
    code = 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); print(p.pid,flush=True); time.sleep(30)'
    # Allow slow Colab CPU process startup before testing process-tree cleanup.
    r = entry.limited_process([sys.executable, "-c", code], log, 3.0)
    assert r["budget_exhausted"]
    try:
        pid = int(log.read_text().strip())
    except (OSError, ValueError) as exc:
        raise AssertionError("child PID was not recorded") from exc
    # Some OSes briefly retain a killed child as zombie; it must not be running.
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert not state or state.startswith("Z")


def test_coordinator_sigterm_cleans_separate_worker_session(tmp_path):
    log = tmp_path / "cancel-child.log"
    child = "import os,time;print(os.getpid(),flush=True);time.sleep(30)"
    code = (
        "import signal,sys; from scripts.train_research_answer_grpo_v1 import limited_process,cancel_campaign; "
        "signal.signal(signal.SIGTERM,cancel_campaign); "
        f"limited_process([sys.executable,'-c',{child!r}],{str(log)!r},30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not log.exists() or not log.read_text().strip()
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert log.exists() and log.read_text().strip()
        try:
            pid = int(log.read_text().strip())
        except (OSError, ValueError) as exc:
            raise AssertionError("worker PID missing") from exc
        process.terminate()
        process.communicate(timeout=5)
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("mode", ["complete", "timeout", "failure"])
def test_campaign_matches_sources_hard_caps_and_never_retries(
    tmp_path, monkeypatch, mode
):
    clock = [100.0]
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock[0])
    plan = {
        "config": CONFIG,
        "source_hashes": {"training": "same"},
        "state": "CPU-test-plan",
    }
    calls = []

    def run(cmd, log_path, timeout):
        calls.append((cmd, timeout))
        clock[0] += timeout if mode == "timeout" else 10
        directory = Path(cmd[cmd.index("--output") + 1])
        directory.mkdir()
        state = {
            "source_hashes": plan["source_hashes"],
            "initial_adapter_tensor_sha256": "same-SFT30",
            "planned_batches_completed": 60,
            "validation_points_completed": [20, 40, 60],
        }
        (directory / "result.json").write_text(json.dumps(state))
        (directory / "plan.json").write_text(json.dumps(plan))
        return {
            "returncode": 7 if mode == "failure" else 0,
            "budget_exhausted": mode == "timeout",
            "wall_seconds": timeout,
        }

    monkeypatch.setattr(entry, "limited_process", run)
    args = SimpleNamespace(
        output=tmp_path / "run",
        base_dir=tmp_path / "base",
        sft_run_dir=tmp_path / "sft",
    )
    if mode == "failure":
        with pytest.raises(RuntimeError):
            entry.campaign(args, plan)
        assert len(calls) == 1
    else:
        result = entry.campaign(args, plan)
        assert len(calls) == 3 and all(t <= 9000 for _, t in calls)
        assert len({c[c.index("--seed") + 1] for c, _ in calls}) == 3
        assert not result["stable_improvement_established"] and not result["test_used"]
        if mode == "timeout":
            assert (
                sum(t for _, t in calls) == 27000
                and result["state"] == "campaign-budget-or-signal-limited"
            )
        else:
            assert result["state"] == "campaign-completed-awaiting-analysis"


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(
            torch.arange(35, dtype=torch.float32).reshape(7, 5) / 100
        )
        self.lora_B = torch.nn.Parameter(
            torch.arange(35, dtype=torch.float32).reshape(5, 7) / 200
        )

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(
            logits=torch.nn.functional.one_hot(input_ids, 7).float()
            @ self.lora_A
            @ self.lora_B
        )


def test_new_rewards_drive_unchanged_real_cpu_kernel_and_zero_groups():
    model = TinyPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.0)
    fp = tensor_fingerprint(lora_parameters(model), torch)
    groups = []
    for gi in range(2):
        members = []
        for mi in range(4):
            t = {
                "prompt_ids": [1, 2],
                "completion_ids": [3 + mi % 2],
                "raw_text": "{}",
                "finish_reason": "stop",
                "policy_tensor_sha256": fp,
            }
            with torch.no_grad():
                lp = token_logps(model, t, torch, device="cpu").tolist()
            t.update(old_logps=lp, teacher_logps=lp)
            members.append(
                {
                    "sample_id": f"train-{gi}",
                    "reward": 1.2 if mi != 1 else 0.2,
                    "turns": [t],
                }
            )
        groups.append(members)
    r = update_batch(model, opt, groups, CONFIG, torch, device="cpu")
    assert r["optimizer_steps"] == 1 and r["changed_tensors"] > 0 and r["grad_norm"] > 0
    for group in groups:
        for member in group:
            member["reward"] = 1.2
    zero = update_batch(model, opt, groups, CONFIG, torch, device="cpu")
    assert zero["optimizer_steps"] == 0 and not zero["effective"]


def test_new_rollout_replaces_trace_reward_without_dropping_actions(records):
    record = next(r for r in records if r["reference_depth"] == 1)
    expert = build_expert(record)
    fn = expert["messages"][2]["tool_calls"][0]["function"]
    outputs = [
        "<tool_call>" + json.dumps(fn) + "</tool_call>",
        '{"unit":"WRONG_UNIT","value":0}',
    ]
    calls = []

    def generate(*args):
        raw = outputs[len(calls)]
        calls.append(1)
        return {
            "prompt_ids": [1, 2],
            "completion_ids": [3],
            "old_logps": [-1.0],
            "raw_text": raw,
            "finish_reason": "stop",
        }

    payload = answer_rollout(record, 37, generate=generate)
    assert (
        len(payload["turns"]) == 2
        and payload["reward_breakdown"]["version"] == "reward-answer-dominant-v1"
    )
    assert (
        not payload["reward_breakdown"]["answer_correct"]
        and payload["reward"] <= 0.200001
    )

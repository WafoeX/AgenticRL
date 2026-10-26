#!/usr/bin/env python3
"""Answer-dominant GRPO v1: CPU preflight default, explicit 3-seed GPU opt-in.
No SFT updates, old pilot run, test access, automatic retry or resume.
"""

import argparse
import asyncio
import contextlib
import gc
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from scripts.eval_research_baselines import REVISION, VERSIONS
from scripts.pilot_research_grpo_step30 import SELECTED_FP, Sampler, selected_source
from scripts.pilot_research_sft import load_base
from scripts.research_answer_grpo_v1 import (
    answer_rollout,
    choose_candidate,
    member_seed,
    training_schedule,
    update_batch,
    validate_config,
)
from scripts.train_research_sft import append_json, spec_for, validate_candidate
from src.research_environment import TOOL_SCHEMAS
from src.research_grpo_runtime import verify_fresh_optimizer
from src.research_model_identity import file_hash, resolve_identity, verify_unchanged
from src.research_registration import load_registered_development
from src.research_schema import load_json, model_messages
from src.research_sft_pilot import tensor_fingerprint
from src.research_sft_run import lora_parameters, save_resume, write_json

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = list((ROOT / "src").glob("*.py")) + [
        ROOT / n
        for n in [
            "scripts/research_answer_grpo_v1.py",
            "scripts/train_research_answer_grpo_v1.py",
            "scripts/research_grpo_native.py",
            "scripts/pilot_research_grpo_step30.py",
            "scripts/train_research_sft.py",
            "scripts/pilot_research_sft.py",
            "scripts/eval_research_baselines.py",
            "configs/research/grpo_answer_three_seed_v1.json",
            "requirements-sft-pilot.txt",
        ]
    ]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def prepare(args):
    config = validate_config(
        load_json(ROOT / "configs/research/grpo_answer_three_seed_v1.json")
    )
    spec, identity = selected_source(args.base_dir, args.sft_run_dir)
    training, train_source = load_registered_development(ROOT, split="train")
    validation, val_source = load_registered_development(ROOT, split="validation")
    schedule = training_schedule(training, config)
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        identity["base_path"],
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    for batch in schedule:
        for record in batch:
            ids = tokenizer.apply_chat_template(
                model_messages(record),
                tools=TOOL_SCHEMAS,
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(ids) > config["policy"]["max_prompt_tokens"]:
                raise ValueError(
                    "initial train prompt over budget; no dropping/replacement"
                )
    selected = load_json(args.sft_run_dir / "selection.json")["best"]
    if (
        selected["answer_correct"] != 128
        or selected["dynamic_valid"] != 75
        or not selected["comparison_valid"]
    ):
        raise ValueError("reviewed SFT30 baseline scores differ")
    baseline = {
        "planned_batch": 0,
        "effective_updates": 0,
        "answer_correct": 128,
        "dynamic_valid": 75,
        "scenarios": 180,
        "comparison_valid": True,
        "model_spec": spec,
        "identity": identity,
        "source": "reused-reviewed-SFT30-not-reevaluated",
    }
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    protocol["modes"] = ["deterministic"]
    plan = {
        "version": config["version"],
        "state": "CPU-preflight-no-official-model-forward",
        "config": config,
        "model_spec": spec,
        "identity": identity,
        "baseline": baseline,
        "train_source": train_source,
        "validation_source": val_source,
        "source_hashes": source_hashes(),
        "schedule": [
            [
                {"sample_id": r["sample_id"], "depth": r["reference_depth"]}
                for r in batch
            ]
            for batch in schedule
        ],
        "max_rollouts_per_seed": 480,
        "max_rollouts_total": 1440,
        "max_validation_tasks": 1620,
        "max_sampling_requests": 8640,
        "max_sampling_completion_tokens": 2211840,
        "campaign_scope_seconds": 27000,
        "historical_remaining_GPU_budget_verified": False,
        "test_used": False,
        "SFT_training_performed": False,
    }
    return plan, tokenizer, schedule, validation, protocol


def cancel_campaign(signum, _frame):
    # Turn parent SIGTERM into Python unwinding so limited_process kills its
    # separately-sessioned GPU worker before the coordinator exits.
    raise KeyboardInterrupt(f"campaign termination requested by signal {signum}")


def limited_process(command, log_path, timeout):
    """Hard monotonic wait limit; kill ONLY this new worker process group.

    No shell, retry, daemon or hidden GPU process. A kill may leave an incomplete
    NEW checkpoint; only committed.json and complete validation may be selected.
    """
    if timeout <= 0:
        return {"returncode": None, "budget_exhausted": True, "wall_seconds": 0.0}
    started = time.monotonic()
    with Path(log_path).open("x") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=ROOT,
        )
        exhausted = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            exhausted = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return {
        "returncode": code,
        "budget_exhausted": exhausted,
        "wall_seconds": time.monotonic() - started,
    }


def save_snapshot(
    output, planned, effective, model, optimizer, torch, identity, args, config
):
    directory = output / f"batch-{planned:04d}"
    directory.mkdir(exist_ok=False)
    checkpoint = directory / f"checkpoint-{effective}"
    model.save_pretrained(
        str(checkpoint), safe_serialization=True, save_embedding_layers=False
    )
    write_json(
        checkpoint / "research_checkpoint.json",
        {
            "step": effective,
            "base_fingerprint": identity["base_fingerprint"],
            "base_revision": REVISION,
        },
    )
    spec = spec_for(args.base_dir, REVISION, checkpoint, effective)
    final = resolve_identity(spec)
    save_resume(
        directory / "optimizer-state",
        model,
        optimizer,
        torch,
        step=effective,
        contract={
            "version": config["version"],
            "seed": args.seed,
            "parent": SELECTED_FP,
            "planned_batches_completed": planned,
            "config": config,
        },
        cuda=True,
    )
    row = {
        "planned_batch": planned,
        "effective_updates": effective,
        "model_spec": spec,
        "identity": final,
        "adapter_tensor_sha256": tensor_fingerprint(lora_parameters(model), torch),
        "optimizer_resume_CLI_implemented": False,
    }
    write_json(
        directory / "committed.json", row
    )  # marker LAST; incomplete snapshots are never candidates
    return row


async def worker(args):
    plan, tokenizer, schedule, validation, protocol = prepare(args)
    parent = load_json(args.output.parent / "plan.json")
    if parent != plan or args.seed not in plan["config"]["seeds"]:
        raise ValueError("seed/source/config differs from parent campaign")
    config = plan["config"]
    deadline = min(args.deadline, time.time() + config["seed_wall_seconds"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    candidates = [plan["baseline"]]
    state = {
        "version": config["version"],
        "seed": args.seed,
        "state": "loading",
        "planned_batches_completed": 0,
        "effective_updates": 0,
        "candidates": candidates,
        "test_used": False,
        "SFT_training_performed": False,
        "optimizer_restored": False,
        "source_hashes": plan["source_hashes"],
        "initial_identity": plan["identity"],
    }
    model = sampler = torch = None
    begun = time.monotonic()

    def log(row):
        append_json(args.output / "events.jsonl", row)

    try:
        versions = {p: metadata.version(p) for p in VERSIONS}
        if any(versions[p].split("+")[0] != v for p, v in VERSIONS.items()):
            raise ValueError("requires unchanged SFT package versions")
        if time.time() >= deadline:
            raise TimeoutError("seed budget expired before CUDA load")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("one BF16 CUDA GPU required")
        props = torch.cuda.get_device_properties(0)
        if "A100" not in props.name or props.total_memory < 39 * 1024**3:
            raise ValueError("single A100 40GB required")
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        identity = plan["identity"]
        model = peft.PeftModel.from_pretrained(
            load_base(identity, torch, transformers),
            identity["checkpoint_path"],
            is_trainable=True,
            local_files_only=True,
            use_safetensors=True,
        )
        saved = importlib.import_module("safetensors.torch").load_file(
            str(Path(identity["checkpoint_path"]) / "adapter_model.safetensors"),
            device="cpu",
        )
        actual = peft.get_peft_model_state_dict(model)
        if set(actual) != set(saved) or any(
            not torch.equal(actual[k].detach().cpu(), saved[k]) for k in saved
        ):
            raise ValueError("actual initial weights differ from SFT30")
        del actual, saved
        parameters = lora_parameters(model)
        if (
            len(parameters) != 336
            or sum(p.numel() for p in parameters.values()) != 4399104
            or list(model.active_adapters) != ["default"]
        ):
            raise ValueError("LoRA shape/active adapter differs")
        optimizer = torch.optim.AdamW(
            parameters.values(), lr=config["policy"]["learning_rate"], weight_decay=0.0
        )
        verify_fresh_optimizer(optimizer, list(parameters.values()))
        state.update(
            initial_adapter_tensor_sha256=tensor_fingerprint(parameters, torch),
            fresh_optimizer_state_entries=0,
            GPU=props.name,
            versions=versions,
            state="training",
        )
        sampler = Sampler(model, tokenizer, torch, transformers, config["policy"], log)
        write_json(args.output / "result.json", state)
        write_json(
            args.output / "selection.json",
            {"best": plan["baseline"], "candidates": candidates},
        )
        for planned, batch in enumerate(schedule, 1):
            if deadline - time.time() < config["validation_reserve_seconds"]:
                state["state"] = "budget-reserve-stop"
                break
            groups = []
            for gi, record in enumerate(batch):
                members = []
                for mi in range(config["group_size"]):
                    sampler.context = {
                        "planned_batch": planned,
                        "group": gi,
                        "member": mi,
                        "sample_id": record["sample_id"],
                    }
                    payload = answer_rollout(
                        record,
                        member_seed(config, args.seed, planned, gi, mi),
                        generate=sampler,
                    )
                    append_json(
                        args.output / "rollouts.jsonl",
                        {**sampler.context, "payload": payload},
                    )
                    members.append(payload)
                groups.append(members)
            receipt = update_batch(
                model, optimizer, groups, config, torch, device="cuda:0"
            )
            state["effective_updates"] += receipt["optimizer_steps"]
            state["planned_batches_completed"] = planned
            sampler.policy_fp = tensor_fingerprint(parameters, torch)
            log({"kind": "update", "planned_batch": planned, **receipt})
            print(
                f"seed={args.seed} batch={planned}/60 effective={state['effective_updates']} signal={receipt['effective']}",
                flush=True,
            )
            if planned % config["save_every_batches"] == 0:
                snapshot = save_snapshot(
                    args.output,
                    planned,
                    state["effective_updates"],
                    model,
                    optimizer,
                    torch,
                    identity,
                    args,
                    config,
                )
                state["last_committed_snapshot"] = snapshot
            write_json(args.output / "result.json", state)
            if planned in config["validation_batches"]:
                if deadline - time.time() < config["validation_reserve_seconds"]:
                    state["state"] = "budget-before-validation-stop"
                    break
                state["state"] = "validating"
                write_json(args.output / "result.json", state)
                snapshot = state["last_committed_snapshot"]
                row = await validate_candidate(
                    args.output / f"validation-batch-{planned:04d}",
                    snapshot["model_spec"],
                    snapshot["identity"],
                    model,
                    tokenizer,
                    validation,
                    protocol,
                    versions,
                    torch,
                    {"validation_scenarios": 180},
                )
                candidate = {
                    **row,
                    "planned_batch": planned,
                    "effective_updates": state["effective_updates"],
                    "source": "answer-GRPO-v1",
                }
                candidates.append(candidate)
                selected = choose_candidate(candidates)
                write_json(
                    args.output / "selection.json",
                    {
                        "best": selected,
                        "candidates": candidates,
                        "best_model_path": selected["model_spec"]["checkpoint_path"],
                        "seed": args.seed,
                    },
                )
                state["state"] = "training"
                if state["effective_updates"] == 0:
                    state["state"] = "no-signal-stop"
                    break
        else:
            state["state"] = "planned-training-completed-not-effectiveness-claim"
        # Preserve a partial final block; never evaluate at unregistered extra points.
        last = state["planned_batches_completed"]
        if (
            last
            and state.get("last_committed_snapshot", {}).get("planned_batch") != last
        ):
            state["last_committed_snapshot"] = save_snapshot(
                args.output,
                last,
                state["effective_updates"],
                model,
                optimizer,
                torch,
                identity,
                args,
                config,
            )
        selected = choose_candidate(candidates)
        state.update(
            best=selected,
            selected_RL_over_parent=selected["planned_batch"] > 0,
            validation_points_completed=[
                c["planned_batch"] for c in candidates if c["planned_batch"]
            ],
        )
        verify_unchanged(plan["model_spec"], identity)
        if source_hashes() != plan["source_hashes"]:
            raise ValueError("runtime code changed during seed")
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["wall_seconds"] = time.monotonic() - begun
        if sampler is not None:
            state["sampling_cost"] = {
                "requests": sampler.requests,
                "prompt_tokens_returned": sampler.prompt_tokens,
                "completion_tokens_returned": sampler.completion_tokens,
                "pending_usage_may_be_unknown": state["state"] == "failed",
            }
        if torch is not None and torch.cuda.is_available():
            state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(args.output / "result.json", state)
        sampler = model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state


def campaign(args, plan):
    config = plan["config"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    started = time.monotonic()
    deadline = started + config["campaign_wall_seconds"]
    epoch_started = time.time()
    summary = {
        "state": "running",
        "seeds": {},
        "config": config,
        "test_used": False,
        "stable_improvement_established": False,
    }
    write_json(
        args.output / "budget.json",
        {
            "started_epoch": epoch_started,
            "campaign_cap_seconds": 27000,
            "seed_cap_seconds": 9000,
            "accounts_for": "workers including loading/training/saving/validation; excludes setup and idle",
            "historical_remaining_GPU_budget_verified": False,
        },
    )
    write_json(args.output / "result.json", summary)
    try:
        for seed in config["seeds"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                summary["state"] = "campaign-budget-exhausted"
                break
            limit = min(config["seed_wall_seconds"], remaining)
            output = args.output / f"seed-{seed}"
            command = [
                sys.executable,
                "-m",
                "scripts.train_research_answer_grpo_v1",
                "--base-dir",
                str(args.base_dir.resolve()),
                "--sft-run-dir",
                str(args.sft_run_dir.resolve()),
                "--output",
                str(output),
                "--run-gpu",
                "--confirm-three-seed-budget",
                "--seed",
                str(seed),
                "--deadline",
                str(time.time() + limit),
            ]
            print(
                f"Start seed {seed}, cap {limit:.0f}s, log={args.output / f'seed-{seed}.log'}",
                flush=True,
            )
            receipt = limited_process(command, args.output / f"seed-{seed}.log", limit)
            state = (
                load_json(output / "result.json")
                if (output / "result.json").exists()
                else None
            )
            summary["seeds"][str(seed)] = {"process": receipt, "result": state}
            if receipt["budget_exhausted"]:
                summary["seeds"][str(seed)]["state"] = "hard-budget-stop-incomplete"
            elif receipt["returncode"] != 0:
                raise RuntimeError(f"seed {seed} failed; no automatic rerun")
            elif state is None or state["source_hashes"] != plan["source_hashes"]:
                raise ValueError("missing/changed worker source receipt")
            if (output / "plan.json").exists() and load_json(
                output / "plan.json"
            ) != plan:
                raise ValueError("worker data/config/schedule/identity differs")
            write_json(args.output / "result.json", summary)
        else:
            completed = all(
                s["result"] is not None
                and not s["process"]["budget_exhausted"]
                and s["result"]["planned_batches_completed"] == 60
                and s["result"].get("validation_points_completed") == [20, 40, 60]
                for s in summary["seeds"].values()
            )
            summary["state"] = (
                "campaign-completed-awaiting-analysis"
                if completed
                else "campaign-budget-or-signal-limited"
            )
        fingerprints = {
            s["result"].get("initial_adapter_tensor_sha256")
            for s in summary["seeds"].values()
            if s["result"]
        }
        if len(fingerprints - {None}) > 1:
            raise ValueError("seeds did not start from identical loaded SFT30 tensors")
    except BaseException as exc:
        summary.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["campaign_wall_seconds"] = time.monotonic() - started
        write_json(args.output / "result.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--sft-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--confirm-three-seed-budget", action="store_true")
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_gpu != args.confirm_three_seed_budget or (
        args.seed is not None and not args.run_gpu
    ):
        parser.error("requires BOTH explicit GPU/campaign confirmations")
    args.output = args.output.expanduser().resolve()
    for source in [args.base_dir.resolve(), args.sft_run_dir.resolve()]:
        if args.output.is_relative_to(source) or source.is_relative_to(args.output):
            parser.error("output must not overlap immutable SFT/Base")
    if args.seed is not None:
        if args.deadline is None:
            parser.error("internal worker requires parent deadline")
        asyncio.run(worker(args))
        return
    plan, *_ = prepare(args)
    if not args.run_gpu:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "state": plan["state"],
                    "seeds": plan["config"]["seeds"],
                    "test_used": False,
                }
            )
        )
        return
    previous = signal.signal(signal.SIGTERM, cancel_campaign)
    try:
        result = campaign(args, plan)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps({"state": result["state"], "seeds": list(result["seeds"])}))
    if result["state"] != "campaign-completed-awaiting-analysis":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

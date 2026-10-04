#!/usr/bin/env python3
"""Evaluate the latest ART checkpoint using the original rollout/tool/reward loop.

Default mode remains sampled evaluation for old --generations commands.
Deterministic accuracy is one greedy attempt per scenario. Sample mode reports
observed pass@k (at least one successful attempt per scenario), separately from
legacy rollout success rate. Success still means reward > 2, NOT final-answer
correctness. Infrastructure failures are reported, not silently discarded.

Examples:
    python scripts/eval_checkpoint.py --mode deterministic
    python scripts/eval_checkpoint.py --mode sample --generations 8
    python scripts/eval_checkpoint.py --mode both --generations 8

Scenarios are gathered in sequential batches of four by default. This caps
scenario concurrency, not individual rollout concurrency (sample mode: 4*k).
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import art
import yaml
from art.local.backend import LocalBackend

# Keep the existing ART monitor workaround; no checkpoint/backend changes.
_orig_monitor = LocalBackend._monitor_openai_server


async def _safe_monitor(self, *args, **kwargs):
    try:
        await _orig_monitor(self, *args, **kwargs)
    except Exception as e:
        print(
            f"[eval] suppressed _monitor_openai_server crash: {type(e).__name__}: {e}",
            file=sys.stderr,
        )


LocalBackend._monitor_openai_server = _safe_monitor

# Also support direct `python scripts/eval_checkpoint.py` without PYTHONPATH.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.diagnostics import (  # noqa: E402
    SUCCESS_REWARD_THRESHOLD,
    format_diagnostics,
    summarize_by_scenario_type,
    summarize_by_tool_depth,
    summarize_complete_chains,
    summarize_evaluation_groups,
)
from src.reward_v2 import is_complete_tool_chain  # noqa: E402
from src.rewards import compute_reward_by_version, compute_reward_v2, validate_reward_version  # noqa: E402
from src.rollout import rollout  # noqa: E402


async def _recorded_rollout(model, scenario, generation_kwargs, reward_version, record, capture_trace):
    # This wrapper only binds sidecar data to the returned trajectory identity;
    # the actual Agent loop remains exclusively in src.rollout.rollout().
    kwargs: dict[str, Any] = dict(generation_kwargs)
    if reward_version != "v1":
        kwargs["reward_version"] = reward_version
    if capture_trace:
        kwargs["trace_sink"] = record["tool_trace"]
    trajectory = await rollout(model, scenario, **kwargs)
    record["trajectory"] = trajectory
    return trajectory


def _audit_attempt(trajectory, scenario, record, complete_chain_enabled, compare_rewards, save_traces):
    trace = record["tool_trace"]
    calls = [{"name": entry["name"], "arguments": entry["arguments"]} for entry in trace]
    failed = trajectory.metrics.get("failed", 0) == 1
    consistent = len(trace) == trajectory.metrics.get("num_tools", 0)
    audit = {"reward": trajectory.reward, "failed": failed}
    if complete_chain_enabled:
        audit["complete_tool_chain"] = not failed and consistent and is_complete_tool_chain(
            calls, scenario.get("expected_calls", []), observations=trace,
        )
    if compare_rewards:
        comparison: dict[str, Any] = {"v1": None, "v2": None, "excluded_reason": None}
        if failed or not consistent:
            comparison["excluded_reason"] = "infra_failure" if failed else "trace_mismatch"
        else:
            expected = scenario.get("expected_calls", [])
            breakdown = compute_reward_v2(calls, expected, observations=trace)
            comparison["v1"] = compute_reward_by_version(calls, expected, version="v1")
            comparison["v2"] = breakdown.total
            comparison["v2_breakdown"] = asdict(breakdown)
        audit["reward_comparison"] = comparison
    if save_traces:
        audit["tool_trace"] = trace
    return audit


async def _evaluate_mode(
    model, scenarios, current_step, scenarios_path, num_generations,
    *, mode, batch_scenarios, max_tokens, diagnostics_enabled, reward_version,
    complete_chain_enabled, compare_rewards, save_traces,
) -> dict:
    generations = 1 if mode == "deterministic" else num_generations
    generation_kwargs = {
        "temperature": 0.0 if mode == "deterministic" else 1.2,
        "top_p": 1.0 if mode == "deterministic" else 0.95,
        "max_tokens": max_tokens,
        "require_logprobs": False,
    }
    print(f"Running {mode}: {generations} rollout(s)/scenario; {batch_scenarios} scenarios/batch")
    eval_groups = []
    eval_records = []
    capture_trace = complete_chain_enabled or compare_rewards or save_traces
    for start in range(0, len(scenarios), batch_scenarios):
        batch = scenarios[start:start + batch_scenarios]
        groups = []
        batch_records = []
        for scenario in batch:
            records = [{"tool_trace": []} for _ in range(generations)]
            batch_records.append(records)
            groups.append(art.TrajectoryGroup([
                _recorded_rollout(
                    model, scenario, generation_kwargs, reward_version, record, capture_trace,
                ) for record in records
            ]))
        resolved = await art.gather_trajectory_groups(
            groups, pbar_desc=f"eval {mode} [{start + 1}-{start + len(batch)}]",
        )
        if len(resolved) != len(batch):
            raise RuntimeError("Evaluation returned a different number of scenario groups")
        if any(len(group.trajectories) != generations for group in resolved):
            raise RuntimeError("Evaluation returned an incomplete set of rollout attempts")
        # ART may reorder members within a group; bind traces by object identity,
        # not by completion order or reward sorting. Never alter the ART objects.
        for group, records in zip(resolved, batch_records, strict=True):
            by_identity = {id(record["trajectory"]): record for record in records}
            identities = [id(trajectory) for trajectory in group.trajectories]
            if len(set(identities)) != generations or set(identities) != set(by_identity):
                raise RuntimeError("Evaluation trajectories do not match their captured records")
            eval_records.append([by_identity[identity] for identity in identities])
        eval_groups.extend(resolved)

    summary = summarize_evaluation_groups(eval_groups)
    per_scenario_stats = []
    attempt_audits = []
    for si, group in enumerate(eval_groups):
        group_rewards = [t.reward for t in group.trajectories]
        success_count = sum(r > SUCCESS_REWARD_THRESHOLD for r in group_rewards)
        mean = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
        max_reward = max(group_rewards) if group_rewards else 0.0
        print(
            f"  {si + 1:2d}/{len(scenarios)}: mean={mean:+.2f} max={max_reward:+.1f} "
            f"success={success_count}/{len(group_rewards)} | {scenarios[si]['task'][:60]}"
        )
        stats = {
            "scenario_index": si,
            "task": scenarios[si]["task"],
            "mean_reward": mean,
            "max_reward": max_reward,
            "success_count": success_count,
            "total_attempts": len(group_rewards),
        }
        if diagnostics_enabled:
            stats["scenario_id"] = scenarios[si].get("id")
        if capture_trace:
            audits = [
                _audit_attempt(
                    trajectory, scenarios[si], record, complete_chain_enabled, compare_rewards, save_traces,
                )
                for trajectory, record in zip(group.trajectories, eval_records[si], strict=True)
            ]
            attempt_audits.append(audits)
            stats["attempts"] = audits
            if save_traces:
                stats["expected_calls"] = scenarios[si].get("expected_calls", [])
        per_scenario_stats.append(stats)

    metric_name = "Deterministic Accuracy" if mode == "deterministic" else f"Pass@{generations}"
    print(f"\n{'=' * 60}\n  EVALUATION RESULTS — checkpoint step {current_step} ({mode})")
    print(f"  Scenarios evaluated: {len(scenarios)} | Total rollouts: {summary['total_attempts']}")
    print(f"  {metric_name}: {summary['task_success_pct']:.1f}%")
    print(f"  Sampled rollout success rate: {summary['accuracy_pct']:.1f}%" if mode == "sample"
          else f"  Rollout success rate: {summary['accuracy_pct']:.1f}%")
    print(f"  Average Reward (all attempts): {summary['avg_reward']:+.3f}")
    print(f"  Infra failure rate: {summary['diagnostics']['infra_failure_rate']:.1%}")
    print("=" * 60)

    results = {
        "step": current_step,
        "mode": mode,
        "reward_version": reward_version,
        "reward_version_definition": "rollout scoring version; not checkpoint training provenance",
        "scenarios_path": scenarios_path,
        "num_generations": generations,
        "total_scenarios": len(scenarios),
        "total_rollouts": summary["total_attempts"],
        "successes": summary["success_count"],
        # Legacy field keeps the old all-rollout denominator in sample mode.
        "accuracy_pct": summary["accuracy_pct"],
        "avg_reward": summary["avg_reward"],
        "per_scenario": per_scenario_stats,
        "sampling": generation_kwargs,
        "batch_scenarios": batch_scenarios,
        "task_success_count": summary["task_success_count"],
        "infra_failure_rate": summary["diagnostics"]["infra_failure_rate"],
        "valid_avg_reward": summary["diagnostics"]["valid_avg_reward"],
        "success_definition": "reward > 2.0; not final-answer correctness",
        "infra_policy": "failed attempts retained; task success rates are conservative when infra fails",
        "has_infra_failures": bool(summary["diagnostics"]["infra_failed_trajectories"]),
    }
    if mode == "deterministic":
        results["deterministic_accuracy_pct"] = summary["task_success_pct"]
    else:
        results["pass_at_k"] = generations
        results["pass_at_k_pct"] = summary["task_success_pct"]

    if complete_chain_enabled:
        outcomes = [[audit["complete_tool_chain"] for audit in audits] for audits in attempt_audits]
        complete_stats = summarize_complete_chains(scenarios, eval_groups, outcomes)
        results["complete_chain"] = complete_stats
        key = "complete_chain_accuracy_pct" if mode == "deterministic" else "complete_chain_pass_at_k_pct"
        results[key] = complete_stats["overall"]["task_success_pct"]
        print(f"  Complete-chain {metric_name}: {results[key]:.1f}% (not final-answer correctness)")
        if diagnostics_enabled:
            for title, buckets in (
                ("Tool Depth", complete_stats["by_tool_depth"]),
                ("Scenario Type", complete_stats["by_scenario_type"]),
            ):
                print(f"  Complete-chain By {title}:")
                for label, stats in buckets.items():
                    print(f"    {label}: {stats['task_success_pct']:.1f}% "
                          f"({stats['successful_scenarios']}/{stats['total_scenarios']})")
    if compare_rewards:
        comparisons = [audit["reward_comparison"] for audits in attempt_audits for audit in audits]
        available = [item for item in comparisons if item["excluded_reason"] is None]
        count = len(available)
        results["reward_comparison"] = {
            "same_trajectories": True,
            "available_rollouts": count,
            "excluded_rollouts": len(comparisons) - count,
            "v1_avg_reward": sum(item["v1"] for item in available) / count if count else None,
            "v2_avg_reward": sum(item["v2"] for item in available) / count if count else None,
            "note": "Scorer comparison only; not evidence of model improvement. Failed/missing traces excluded.",
        }
        print(f"  Same-trace reward comparison ({count} available): {results['reward_comparison']}")

    if results["has_infra_failures"]:
        print("  WARNING: infrastructure failures affect accuracy/pass@k and all-attempt mean reward.")
    if diagnostics_enabled:
        results["diagnostics"] = summary["diagnostics"]
        results["accuracy_definition"] = "rollout reward > 2.0; all attempts"
        print(format_diagnostics(summary["diagnostics"]))
        for title, key, breakdown in (
            ("Tool Depth", "by_tool_depth", summarize_by_tool_depth(scenarios, eval_groups)),
            ("Scenario Type", "by_scenario_type", summarize_by_scenario_type(scenarios, eval_groups)),
        ):
            results[key] = breakdown
            print(f"  By {title} ({metric_name}, reward>2):")
            for label, stats in breakdown.items():
                stats["metric_name"] = metric_name
                print(
                    f"    {label}: {stats['task_success_pct']:.1f}% "
                    f"({stats['task_success_count']}/{stats['total_scenarios']} scenarios)"
                )
    return results


async def main(
    scenarios_path: str, num_generations: int, *, diagnostics_enabled: bool = True,
    mode: str = "sample", batch_scenarios: int = 4, max_tokens: int = 512,
    deterministic_temperature: float = 0.0, output_path: str = "eval_results.json",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_name: str = "qwen-0.5b-tool-agent", project_name: str = "rl-tool-use",
    reward_version: str = "v1", complete_chain_enabled: bool = False,
    compare_rewards: bool = False, save_traces: bool = False,
) -> dict:
    validate_reward_version(reward_version)
    if mode not in {"deterministic", "sample", "both"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if num_generations < 1 or batch_scenarios < 1 or max_tokens < 1:
        raise ValueError("generations, batch_scenarios, and max_tokens must be positive")
    if mode != "sample" and deterministic_temperature != 0.0:
        raise ValueError("Deterministic evaluation requires deterministic_temperature=0.0")
    with open(scenarios_path) as f:
        scenarios = [json.loads(line) for line in f]
    print(f"Loaded {len(scenarios)} validation scenarios")
    backend = LocalBackend()
    model = art.TrainableModel(name=model_name, project=project_name, base_model=base_model)
    await model.register(backend)
    current_step = await model.get_step()
    print(f"Loaded checkpoint: step {current_step}")

    modes = ("deterministic", "sample") if mode == "both" else (mode,)
    reports = {}
    for eval_mode in modes:
        reports[eval_mode] = await _evaluate_mode(
            model, scenarios, current_step, scenarios_path, num_generations,
            mode=eval_mode, batch_scenarios=batch_scenarios, max_tokens=max_tokens,
            diagnostics_enabled=diagnostics_enabled, reward_version=reward_version,
            complete_chain_enabled=complete_chain_enabled, compare_rewards=compare_rewards,
            save_traces=save_traces,
        )
    results = reports[mode] if mode != "both" else {
        "step": current_step, "mode": "both", "scenarios_path": scenarios_path,
        "deterministic_accuracy_pct": reports["deterministic"]["deterministic_accuracy_pct"],
        "pass_at_k": num_generations, "pass_at_k_pct": reports["sample"]["pass_at_k_pct"],
        "avg_reward": reports["sample"]["avg_reward"],
        "avg_reward_definition": "sample mode; deterministic reward is in evaluations.deterministic",
        "evaluations": reports,
    }
    if mode == "both":
        results["reward_version"] = reward_version
        results["reward_version_definition"] = "rollout scoring version; not checkpoint training provenance"
        if complete_chain_enabled:
            results["complete_chain_accuracy_pct"] = reports["deterministic"]["complete_chain_accuracy_pct"]
            results["complete_chain_pass_at_k_pct"] = reports["sample"]["complete_chain_pass_at_k_pct"]
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_path}")
    return results


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="data/scenarios_val.jsonl")
    parser.add_argument("--generations", type=int, default=8, help="Sample attempts; deterministic always uses 1")
    parser.add_argument("--mode", choices=("deterministic", "sample", "both"), default="sample")
    parser.add_argument("--batch-scenarios", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--config", default=None, help="Optional YAML; missing evaluation fields use defaults")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable diagnostic/type/depth breakdowns")
    parser.add_argument("--reward-version", choices=("v1", "v2"), default=None)
    parser.add_argument("--complete-chain", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compare-rewards", action="store_true", help="Score the same captured calls with v1 and v2")
    parser.add_argument("--save-traces", action="store_true", help="Include per-attempt tool traces in output JSON")
    args = parser.parse_args()
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    evaluation = config.get("evaluation", {})
    identity = config.get("model", {})
    asyncio.run(main(
        args.scenarios, args.generations, mode=args.mode,
        diagnostics_enabled=not args.no_diagnostics,
        batch_scenarios=(args.batch_scenarios if args.batch_scenarios is not None
                         else evaluation.get("batch_scenarios", 4)),
        max_tokens=(args.max_tokens if args.max_tokens is not None
                    else evaluation.get("max_tokens", 512)),
        deterministic_temperature=evaluation.get("deterministic_temperature", 0.0),
        output_path=args.output,
        base_model=identity.get("base", "Qwen/Qwen2.5-0.5B-Instruct"),
        model_name=identity.get("name", "qwen-0.5b-tool-agent"),
        project_name=identity.get("project", "rl-tool-use"),
        reward_version=(args.reward_version if args.reward_version is not None else
                        evaluation.get("reward_version", config.get("training", {}).get("reward_version", "v1"))),
        complete_chain_enabled=(args.complete_chain if args.complete_chain is not None else
                                evaluation.get("complete_chain_enabled", False)),
        compare_rewards=args.compare_rewards,
        save_traces=args.save_traces,
    ))


if __name__ == "__main__":
    cli()

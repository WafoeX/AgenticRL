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
from pathlib import Path

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
    summarize_evaluation_groups,
)
from src.rollout import rollout  # noqa: E402


async def _evaluate_mode(
    model, scenarios, current_step, scenarios_path, num_generations,
    *, mode, batch_scenarios, max_tokens, diagnostics_enabled,
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
    for start in range(0, len(scenarios), batch_scenarios):
        batch = scenarios[start:start + batch_scenarios]
        groups = (
            art.TrajectoryGroup(
                rollout(model, scenario, **generation_kwargs) for _ in range(generations)
            )
            for scenario in batch
        )
        resolved = await art.gather_trajectory_groups(
            groups, pbar_desc=f"eval {mode} [{start + 1}-{start + len(batch)}]",
        )
        if len(resolved) != len(batch):
            raise RuntimeError("Evaluation returned a different number of scenario groups")
        if any(len(group.trajectories) != generations for group in resolved):
            raise RuntimeError("Evaluation returned an incomplete set of rollout attempts")
        eval_groups.extend(resolved)

    summary = summarize_evaluation_groups(eval_groups)
    per_scenario_stats = []
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
) -> dict:
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
            diagnostics_enabled=diagnostics_enabled,
        )
    results = reports[mode] if mode != "both" else {
        "step": current_step, "mode": "both", "scenarios_path": scenarios_path,
        "deterministic_accuracy_pct": reports["deterministic"]["deterministic_accuracy_pct"],
        "pass_at_k": num_generations, "pass_at_k_pct": reports["sample"]["pass_at_k_pct"],
        "avg_reward": reports["sample"]["avg_reward"],
        "avg_reward_definition": "sample mode; deterministic reward is in evaluations.deterministic",
        "evaluations": reports,
    }
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
    ))


if __name__ == "__main__":
    cli()

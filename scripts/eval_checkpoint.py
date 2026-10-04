#!/usr/bin/env python3
"""Evaluate a trained ART checkpoint on the validation set.

Loads from `.art/<project>/models/<name>/checkpoints/<step>/` automatically
via ART's `model.register(backend)` — ART picks up the latest saved
checkpoint for the given model name. Runs the val scenarios through the
same `rollout()` loop used in training, but without any gradient updates.

Reports:
  - Per-scenario mean/max reward and success count
  - Overall accuracy (% of rollouts with reward > 2.0)
  - Overall avg reward across all rollouts

Writes detailed per-scenario results to eval_results.json for the blog post.

Usage:
    python scripts/eval_checkpoint.py
    python scripts/eval_checkpoint.py --scenarios data/scenarios_val.jsonl
    python scripts/eval_checkpoint.py --generations 8

Note: uses the same monkey patch as src/train.py to suppress the
_monitor_openai_server crash that plagued training.
"""
import argparse
import asyncio
import json
import sys

import art
from art.local.backend import LocalBackend

# Apply the same monkey patch as src/train.py: suppress the background
# _monitor_openai_server crash that otherwise kills long-running ART jobs.
# 中文注释：评估也复用训练时的监控补丁，避免 ART 后台监控异常中断长评估。
_orig_monitor = LocalBackend._monitor_openai_server


# 中文注释：吞掉监控线程的已知异常，但把错误写入标准错误便于排查。
async def _safe_monitor(self, *args, **kwargs):
    try:
        await _orig_monitor(self, *args, **kwargs)
    except Exception as e:
        print(
            f"[eval] suppressed _monitor_openai_server crash: {type(e).__name__}: {e}",
            file=sys.stderr,
        )


LocalBackend._monitor_openai_server = _safe_monitor

# Import after monkey patch so train.py's patch logic (if applied) is consistent.
from src.diagnostics import (  # noqa: E402
    format_diagnostics,
    summarize_by_scenario_type,
    summarize_trajectory_groups,
)
from src.rollout import rollout  # noqa: E402


# 中文注释：加载最新 ART 检查点，对每个验证场景执行多次 rollout 并汇总统计。
async def main(
    scenarios_path: str, num_generations: int, *, diagnostics_enabled: bool = True,
) -> None:
    with open(scenarios_path) as f:
        scenarios = [json.loads(line) for line in f]

    print(f"Loaded {len(scenarios)} validation scenarios")
    print(f"Running {num_generations} rollouts per scenario")

    backend = LocalBackend()
    model = art.TrainableModel(
        name="qwen-0.5b-tool-agent",
        project="rl-tool-use",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    await model.register(backend)

    current_step = await model.get_step()
    print(f"Loaded checkpoint: step {current_step}")
    print()

    # Build one TrajectoryGroup per scenario (same pattern as training).
    groups = (
        art.TrajectoryGroup(rollout(model, scenario) for _ in range(num_generations))
        for scenario in scenarios
    )

    eval_groups = await art.gather_trajectory_groups(
        groups,
        pbar_desc="eval",
    )

# 中文注释：下面把每个场景的奖励聚合成总准确率、平均奖励和明细列表。
    # Aggregate results.
    total_trajs = 0
    success_trajs = 0
    total_reward = 0.0
    per_scenario_stats = []

    for si, group in enumerate(eval_groups):
        group_trajs = list(group.trajectories)
        group_rewards = [t.reward for t in group_trajs]
        group_success = sum(1 for r in group_rewards if r > 2.0)

        total_trajs += len(group_trajs)
        success_trajs += group_success
        total_reward += sum(group_rewards)

        mean = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
        max_r = max(group_rewards) if group_rewards else 0.0
        task_preview = scenarios[si]["task"][:60]
        print(
            f"  {si+1:2d}/{len(scenarios)}: "
            f"mean={mean:+.2f} max={max_r:+.1f} "
            f"success={group_success}/{len(group_trajs)} | {task_preview}"
        )

        per_scenario_stats.append(
            {
                "scenario_index": si,
                "task": scenarios[si]["task"],
                "mean_reward": mean,
                "max_reward": max_r,
                "success_count": group_success,
                "total_attempts": len(group_trajs),
            }
        )

    accuracy = (success_trajs / total_trajs * 100) if total_trajs else 0.0
    avg_reward = total_reward / total_trajs if total_trajs else 0.0

    print()
    print("=" * 60)
    print(f"  EVALUATION RESULTS — checkpoint step {current_step}")
    print("=" * 60)
    print(f"  Scenarios evaluated:    {len(scenarios)}")
    print(f"  Rollouts per scenario:  {num_generations}")
    print(f"  Total rollouts:         {total_trajs}")
    print(f"  Successes (reward>2.0): {success_trajs}")
    print(f"  Accuracy:               {accuracy:.1f}%")
    print(f"  Avg reward:             {avg_reward:+.3f}")
    print("=" * 60)

    results = {
        "step": current_step,
        "scenarios_path": scenarios_path,
        "num_generations": num_generations,
        "total_scenarios": len(scenarios),
        "total_rollouts": total_trajs,
        "successes": success_trajs,
        "accuracy_pct": accuracy,
        "avg_reward": avg_reward,
        "per_scenario": per_scenario_stats,
    }
    if diagnostics_enabled:
        diagnostics = summarize_trajectory_groups(eval_groups)
        by_scenario_type = summarize_by_scenario_type(scenarios, eval_groups)
        print(format_diagnostics(diagnostics))
        print("  By Scenario Type (sampled rollout success, reward>2; NOT task accuracy/pass@N):")
        for scenario_type, stats in by_scenario_type.items():
            print(
                f"    {scenario_type}: {stats['accuracy_pct']:.1f}% "
                f"({stats['success_count']}/{stats['total_attempts']})"
            )
        results["diagnostics"] = diagnostics
        results["by_scenario_type"] = by_scenario_type
        results["accuracy_definition"] = "sampled rollout reward > 2.0; all attempts"
        for scenario, stats in zip(scenarios, per_scenario_stats, strict=True):
            stats["scenario_id"] = scenario.get("id")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="data/scenarios_val.jsonl")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable read-only diagnostics")
    args = parser.parse_args()
    asyncio.run(main(
        args.scenarios, args.generations, diagnostics_enabled=not args.no_diagnostics,
    ))

# src/train.py
"""Main GRPO training loop using OpenPipe ART.

Usage:
    python -m src.train                    # default config
    python -m src.train --steps 10         # quick test
    python -m src.train --config configs/train_config.yaml
"""
# 中文注释：训练入口负责加载场景、注册 ART 模型、收集轨迹并调用 GRPO 更新。
import argparse
import asyncio
import json
from typing import TypedDict

import art
import yaml
from art.local.backend import LocalBackend

from src.backend_lifecycle import install_monitor_workaround, managed_backend
from src.diagnostics import format_diagnostics, summarize_trajectory_groups
from src.rewards import validate_reward_version
from src.rollout import rollout


class _RewardOptions(TypedDict, total=False):
    reward_version: str


# Preserve the existing health-check workaround, shared with evaluation.
# The wrapper also tracks monitors for explicit, backend-scoped teardown.
install_monitor_workaround(LocalBackend)


# 中文注释：训练函数按照场景分组收集多条 rollout，让 GRPO 在组内计算相对优势。
async def train(
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    project_name: str = "rl-tool-use",
    model_name: str = "qwen-0.5b-tool-agent",
    num_steps: int = 50,
    rollouts_per_step: int = 96,
    num_generations: int = 8,
    learning_rate: float = 5e-5,
    train_scenarios_path: str = "data/scenarios_train.jsonl",
    val_scenarios_path: str = "data/scenarios_val.jsonl",
    diagnostics_enabled: bool = True,
    reward_version: str = "v1",
):
    validate_reward_version(reward_version)
    reward_kwargs: _RewardOptions = {} if reward_version == "v1" else {"reward_version": reward_version}
    print(f"Reward version: {reward_version}")
# 中文注释：训练和验证数据以 JSONL 逐行读取，保持数据文件格式简单可审计。
    with open(train_scenarios_path) as f:
        train_scenarios = [json.loads(line) for line in f]
    with open(val_scenarios_path) as f:
        val_scenarios = [json.loads(line) for line in f]

    print(f"Loaded {len(train_scenarios)} train, {len(val_scenarios)} val scenarios")

    # GRPO requires G rollouts per prompt so we can compute relative advantages
    # within a group. rollouts_per_step is the total batch; num_generations is G.
    if rollouts_per_step % num_generations != 0:
        raise ValueError(
            f"rollouts_per_step ({rollouts_per_step}) must be divisible by "
            f"num_generations ({num_generations}) for GRPO"
        )
    scenarios_per_step = rollouts_per_step // num_generations
    print(
        f"GRPO config: {scenarios_per_step} scenarios/step × "
        f"{num_generations} rollouts/scenario = {rollouts_per_step} rollouts/step"
    )

    async with managed_backend(LocalBackend()) as backend:
        model = art.TrainableModel(
            name=model_name,
            project=project_name,
            base_model=base_model,
        )
        await model.register(backend)

        start_step = await model.get_step()
        print(f"Starting from step {start_step}")

        for step in range(start_step, num_steps):
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{num_steps}")
            print(f"{'='*60}")

            # Pick unique scenarios for this step — one per group
            step_scenarios = []
            for i in range(scenarios_per_step):
                idx = (step * scenarios_per_step + i) % len(train_scenarios)
                step_scenarios.append(train_scenarios[idx])

            # Each group = num_generations rollouts of the SAME scenario.
            # This gives GRPO the within-group reward variance it needs.
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(model, scenario, **reward_kwargs) for _ in range(num_generations)
                    )
                    for scenario in step_scenarios
                ),
                pbar_desc=f"step {step + 1}",
            )

            rewards = [t.reward for g in train_groups for t in g.trajectories]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            tools_used = [t.metrics.get("num_tools", 0) for g in train_groups for t in g.trajectories]
            avg_tools = sum(tools_used) / len(tools_used) if tools_used else 0

            # Per-group variance — critical diagnostic. If all groups have std=0,
            # GRPO has no signal and skips training. We need at least some groups
            # with variance for the model to learn.
            groups_with_variance = 0
            for gi, g in enumerate(train_groups):
                group_rewards = [t.reward for t in g.trajectories]
                if not group_rewards:
                    continue
                g_mean = sum(group_rewards) / len(group_rewards)
                g_std = (sum((r - g_mean) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
                if g_std > 0:
                    groups_with_variance += 1
                task_preview = step_scenarios[gi]["task"][:50] if gi < len(step_scenarios) else "?"
                print(
                    f"  group {gi}: mean={g_mean:+.2f} std={g_std:.3f} "
                    f"min={min(group_rewards):+.1f} max={max(group_rewards):+.1f} "
                    f"| {task_preview}"
                )
            print(
                f"  Avg reward: {avg_reward:.3f} | Avg tools/rollout: {avg_tools:.1f} "
                f"| groups with variance: {groups_with_variance}/{len(train_groups)}"
            )

            # Phase 1 is observation only: the original groups/rewards still go to GRPO.
            if diagnostics_enabled:
                print(format_diagnostics(summarize_trajectory_groups(train_groups)))

            # Existing checkpoint retention policy is unchanged.
            await model.delete_checkpoints()
            result = await backend.train(model, train_groups, learning_rate=learning_rate)
            await model.log(train_groups, metrics=result.metrics, step=result.step, split="train")

            if (step + 1) % 5 == 0:
                print("  Running validation...")
                # For val: one group per scenario, num_generations rollouts each
                val_groups = await art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(model, scenario, **reward_kwargs) for _ in range(num_generations)
                        )
                        for scenario in val_scenarios
                    ),
                    pbar_desc="validation",
                )
                val_rewards = [t.reward for g in val_groups for t in g.trajectories]
                val_avg = sum(val_rewards) / len(val_rewards) if val_rewards else 0
                print(f"  Val avg reward: {val_avg:.3f}")
                if diagnostics_enabled:
                    print(format_diagnostics(summarize_trajectory_groups(val_groups)))
                await model.log(val_groups, step=result.step, split="val")

        print(f"\nTraining updates complete! Target step: {num_steps}; closing backend...", flush=True)

    print(f"Training complete! Backend resources released. Target step: {num_steps}", flush=True)


# 中文注释：命令行层把配置文件和命令行覆盖项汇总后启动异步训练。
def main():
    parser = argparse.ArgumentParser(description="Train tool-use agent with GRPO")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable read-only diagnostics")
    parser.add_argument("--reward-version", choices=("v1", "v2"), default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        kwargs = {
            "base_model": config["model"]["base"],
            "model_name": config["model"]["name"],
            "project_name": config["model"]["project"],
            "num_steps": config["training"]["num_steps"],
            "rollouts_per_step": config["training"]["rollouts_per_step"],
            "num_generations": config["training"]["num_generations"],
            "learning_rate": config["training"]["learning_rate"],
            "diagnostics_enabled": config["training"].get("diagnostics_enabled", True),
            "reward_version": config["training"].get("reward_version", "v1"),
        }
    if args.steps:
        kwargs["num_steps"] = args.steps
    if args.rollouts:
        kwargs["rollouts_per_step"] = args.rollouts
    if args.lr:
        kwargs["learning_rate"] = args.lr
    if args.model:
        kwargs["base_model"] = args.model

    if args.no_diagnostics:
        kwargs["diagnostics_enabled"] = False
    if args.reward_version is not None:
        kwargs["reward_version"] = args.reward_version

    asyncio.run(train(**kwargs))


if __name__ == "__main__":
    main()

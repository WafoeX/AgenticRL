# src/evaluate.py
"""Evaluate RL-trained model vs prompted baseline.

Usage:
    python -m src.evaluate --trained-url http://localhost:8000/v1 --baseline-url http://localhost:8001/v1
"""
import asyncio
import argparse
import json
# 中文注释：评估结果使用 dataclass 表达，便于汇总准确率、平均奖励和逐场景信息。
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from src.tools import TOOL_SCHEMAS, execute_tool
from src.rewards import compute_reward


@dataclass
# 中文注释：单个场景结果记录任务、奖励、调用过的工具、轮数和是否成功。
class EvalResult:
    task: str
    reward: float
    tools_called: list[str]
    num_turns: int
    success: bool


@dataclass
# 中文注释：报告对象聚合多个场景，并提供准确率、平均奖励和文本摘要。
class EvalReport:
    model_name: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results) * 100

    @property
    def avg_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reward for r in self.results) / len(self.results)

    def summary(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"Model: {self.model_name}\n"
            f"Scenarios: {len(self.results)}\n"
            f"Accuracy: {self.accuracy:.1f}%\n"
            f"Avg Reward: {self.avg_reward:.2f}\n"
            f"{'='*50}"
        )


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use the provided tools to answer the user's question. "
    "Call tools as needed, then provide your final answer. "
    "Be precise with tool arguments."
)
MAX_EVAL_TURNS = 6


# 中文注释：评估单个场景时关闭采样，沿用工具执行循环但不进行梯度更新。
async def evaluate_single(
    client: AsyncOpenAI,
    model_id: str,
    scenario: dict,
) -> EvalResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario["task"]},
    ]
    tools_called = []
    actual_calls = []

    for turn in range(MAX_EVAL_TURNS):
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as e:
            print(f"  Error on '{scenario['task'][:40]}': {e}")
            break

        choice = response.choices[0]
        if not choice.message.tool_calls:
            messages.append({"role": "assistant", "content": choice.message.content or ""})
            break

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tools_called.append(tool_name)
            actual_calls.append({"name": tool_name, "arguments": tool_args})
            result = execute_tool(tool_name, tool_args)

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tc.function.arguments},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    expected_calls = scenario.get("expected_calls", [])
    reward = compute_reward(actual_calls, expected_calls)

    return EvalResult(
        task=scenario["task"],
        reward=reward,
        tools_called=tools_called,
        num_turns=len(tools_called),
        success=reward > 2.0,
    )


# 中文注释：顺序评估一组场景并实时打印每个场景的成功状态。
async def evaluate_model(
    base_url: str, api_key: str, model_id: str, model_name: str, scenarios: list[dict],
) -> EvalReport:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    report = EvalReport(model_name=model_name)

    for i, scenario in enumerate(scenarios):
        result = await evaluate_single(client, model_id, scenario)
        report.results.append(result)
        status = "OK" if result.success else "FAIL"
        print(f"  [{i+1}/{len(scenarios)}] {status} | R={result.reward:+.1f} | {scenario['task'][:50]}")

    return report


# 中文注释：主评估流程可同时比较 RL 模型和提示基线，并保存 JSON 结果。
async def main_eval(
    scenarios_path: str,
    trained_url: str,
    trained_model: str,
    baseline_url: str | None,
    baseline_model: str | None,
):
    with open(scenarios_path) as f:
        scenarios = [json.loads(line) for line in f]

    print(f"Evaluating {len(scenarios)} scenarios\n")

    print("--- RL-Trained Model ---")
    trained_report = await evaluate_model(
        base_url=trained_url, api_key="not-needed",
        model_id=trained_model, model_name=f"RL-Trained ({trained_model})",
        scenarios=scenarios,
    )
    print(trained_report.summary())

    if baseline_url and baseline_model:
        print("\n--- Prompted Baseline ---")
        baseline_report = await evaluate_model(
            base_url=baseline_url, api_key="not-needed",
            model_id=baseline_model, model_name=f"Prompted ({baseline_model})",
            scenarios=scenarios,
        )
        print(baseline_report.summary())

        delta = trained_report.accuracy - baseline_report.accuracy
        print(f"\n{'='*50}")
        print(f"COMPARISON: RL-Trained vs Prompted")
        print(f"  RL-Trained: {trained_report.accuracy:.1f}%")
        print(f"  Prompted:   {baseline_report.accuracy:.1f}%")
        print(f"  Delta:      {delta:+.1f}%")
        print(f"{'='*50}")

    output = {
        "trained": {
            "accuracy": trained_report.accuracy,
            "avg_reward": trained_report.avg_reward,
            "results": [
                {"task": r.task, "reward": r.reward, "success": r.success, "tools": r.tools_called}
                for r in trained_report.results
            ],
        },
    }
    with open("data/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to data/eval_results.json")


# 中文注释：解析评估数据集、模型地址和模型名等命令行参数。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="data/scenarios_val.jsonl")
    parser.add_argument("--trained-url", default="http://localhost:8000/v1")
    parser.add_argument("--trained-model", default="qwen-0.5b-tool-agent")
    parser.add_argument("--baseline-url", default=None)
    parser.add_argument("--baseline-model", default=None)
    args = parser.parse_args()

    asyncio.run(main_eval(
        scenarios_path=args.scenarios,
        trained_url=args.trained_url,
        trained_model=args.trained_model,
        baseline_url=args.baseline_url,
        baseline_model=args.baseline_model,
    ))


if __name__ == "__main__":
    main()

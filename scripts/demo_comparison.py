# scripts/demo_comparison.py
"""Side-by-side demo: prompted agent vs RL-trained agent.

For the 1-minute demo video (0:25-0:45 segment).

Usage:
    python scripts/demo_comparison.py \
        --trained-url http://localhost:8000/v1 \
        --trained-model qwen-0.5b-tool-agent \
        --baseline-url http://localhost:8001/v1 \
        --baseline-model Qwen/Qwen2.5-7B-Instruct
"""
import asyncio
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate_single, EvalResult
from src.tools import TOOL_SCHEMAS, execute_tool
from openai import AsyncOpenAI

# 中文注释：演示任务覆盖两步和三步工具链，用于直观比较两个模型的工具调用能力。
DEMO_TASKS = [
    {
        "task": "How old was Einstein in 2024?",
        "expected_calls": [
            {"name": "search", "arguments": {"query": "Einstein birth year"}},
            {"name": "calculator", "arguments": {"expression": "2024-1879"}},
        ],
        "label": "2-step: Search + Calculate",
    },
    {
        "task": "What is the temperature in Tokyo in Fahrenheit?",
        "expected_calls": [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "convert_units", "arguments": {"value": 24, "from_unit": "celsius", "to_unit": "fahrenheit"}},
        ],
        "label": "2-step: Weather + Convert",
    },
    {
        "task": "What is France's population density in people per square mile?",
        "expected_calls": [
            {"name": "lookup_db", "arguments": {"table": "countries", "key": "France"}},
            {"name": "calculator", "arguments": {"expression": "67750000/551695"}},
            {"name": "convert_units", "arguments": {"value": 122.8, "from_unit": "km", "to_unit": "miles"}},
        ],
        "label": "3-step: Lookup + Calculate + Convert",
    },
]


# 中文注释：把单次结果格式化为带颜色的通过/失败行。
def print_result(label: str, model_name: str, result: EvalResult):
    status = "\033[92mPASS\033[0m" if result.success else "\033[91mFAIL\033[0m"
    print(f"  {model_name}: {status} | Reward: {result.reward:+.1f} | Tools: {' -> '.join(result.tools_called) or 'none'}")


# 中文注释：分别请求训练模型和提示基线，然后统计各自在固定演示任务上的胜场。
async def run_demo(trained_url: str, trained_model: str, baseline_url: str, baseline_model: str):
    trained_client = AsyncOpenAI(base_url=trained_url, api_key="x")
    baseline_client = AsyncOpenAI(base_url=baseline_url, api_key="x")

    trained_wins = 0
    baseline_wins = 0

    print("\n" + "="*60)
    print("  RL-TRAINED (0.5B) vs PROMPTED (7B) — TOOL USE SHOWDOWN")
    print("="*60 + "\n")

    for task in DEMO_TASKS:
        print(f"\n--- {task['label']} ---")
        print(f"Task: \"{task['task']}\"\n")

        trained_result = await evaluate_single(trained_client, trained_model, task)
        baseline_result = await evaluate_single(baseline_client, baseline_model, task)

        print_result(task["label"], "RL-Trained 0.5B", trained_result)
        print_result(task["label"], "Prompted 7B   ", baseline_result)

        if trained_result.success:
            trained_wins += 1
        if baseline_result.success:
            baseline_wins += 1

    total = len(DEMO_TASKS)
    print(f"\n{'='*60}")
    print(f"  FINAL SCORE")
    print(f"  RL-Trained 0.5B:  {trained_wins}/{total} ({trained_wins/total*100:.0f}%)")
    print(f"  Prompted 7B:      {baseline_wins}/{total} ({baseline_wins/total*100:.0f}%)")
    print(f"{'='*60}\n")


# 中文注释：命令行入口配置两个 OpenAI 兼容服务的地址和模型名。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-url", default="http://localhost:8000/v1")
    parser.add_argument("--trained-model", default="qwen-0.5b-tool-agent")
    parser.add_argument("--baseline-url", default="http://localhost:8001/v1")
    parser.add_argument("--baseline-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    asyncio.run(run_demo(args.trained_url, args.trained_model, args.baseline_url, args.baseline_model))


if __name__ == "__main__":
    main()

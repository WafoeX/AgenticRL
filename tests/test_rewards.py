import pytest
from src.rewards import compute_reward, parse_tool_calls_from_messages


# 中文注释：测试从单轮、多轮和无工具消息中提取调用。
class TestParseToolCalls:
    def test_single_tool_call(self):
        messages = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "calculator", "arguments": '{"expression": "3+5"}'}}
            ]}
        ]
        calls = parse_tool_calls_from_messages(messages)
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"
        assert calls[0]["arguments"] == {"expression": "3+5"}

    def test_multiple_turns(self):
        messages = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "Einstein"}'}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "calculator", "arguments": '{"expression": "2024-1879"}'}}
            ]},
        ]
        calls = parse_tool_calls_from_messages(messages)
        assert len(calls) == 2
        assert calls[0]["name"] == "search"
        assert calls[1]["name"] == "calculator"

    def test_no_tool_calls(self):
        messages = [{"role": "assistant", "content": "I don't know."}]
        calls = parse_tool_calls_from_messages(messages)
        assert len(calls) == 0


# 中文注释：测试完美匹配、参数错误、工具错误、空预测和部分匹配的奖励范围。
class TestComputeReward:
    def test_perfect_match(self):
        predicted = [{"name": "calculator", "arguments": {"expression": "3+5"}}]
        ground_truth = [{"name": "calculator", "arguments": {"expression": "3+5"}}]
        reward = compute_reward(predicted, ground_truth)
        assert reward >= 3.5

    def test_right_tool_wrong_args(self):
        predicted = [{"name": "calculator", "arguments": {"expression": "3*5"}}]
        ground_truth = [{"name": "calculator", "arguments": {"expression": "3+5"}}]
        reward = compute_reward(predicted, ground_truth)
        assert 0.0 < reward < 3.0

    def test_wrong_tool(self):
        predicted = [{"name": "search", "arguments": {"query": "3+5"}}]
        ground_truth = [{"name": "calculator", "arguments": {"expression": "3+5"}}]
        reward = compute_reward(predicted, ground_truth)
        assert reward < 0.0

    def test_no_predictions(self):
        predicted = []
        ground_truth = [{"name": "calculator", "arguments": {"expression": "3+5"}}]
        reward = compute_reward(predicted, ground_truth)
        assert reward <= -2.0

    def test_multi_step_correct(self):
        predicted = [
            {"name": "search", "arguments": {"query": "Einstein birth year"}},
            {"name": "calculator", "arguments": {"expression": "2024-1879"}},
        ]
        ground_truth = [
            {"name": "search", "arguments": {"query": "Einstein birth year"}},
            {"name": "calculator", "arguments": {"expression": "2024-1879"}},
        ]
        reward = compute_reward(predicted, ground_truth)
        assert reward >= 3.5

    def test_multi_step_partial(self):
        predicted = [
            {"name": "search", "arguments": {"query": "Einstein"}},
            {"name": "get_weather", "arguments": {"city": "Paris"}},
        ]
        ground_truth = [
            {"name": "search", "arguments": {"query": "Einstein birth year"}},
            {"name": "calculator", "arguments": {"expression": "2024-1879"}},
        ]
        reward = compute_reward(predicted, ground_truth)
        assert -2.0 < reward < 2.0

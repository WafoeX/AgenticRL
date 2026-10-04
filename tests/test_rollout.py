import asyncio
import json
import sys
import types
import pytest
from unittest.mock import MagicMock, patch

# --- Stub `art` before importing rollout ---

art_stub = types.ModuleType("art")


# 中文注释：测试替身模拟 ART Trajectory，避免单元测试依赖 GPU 和真实 ART。
class _Trajectory:
    def __init__(self, messages_and_choices, reward, metrics):
        self.messages_and_choices = messages_and_choices
        self.reward = reward
        self.metrics = metrics


art_stub.Trajectory = _Trajectory
art_stub.Model = object
sys.modules.setdefault("art", art_stub)

# Also stub `openai` so rollout.py doesn't fail at import time if not installed
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.AsyncOpenAI = MagicMock()

    class _StubBadRequestError(Exception):
        pass

    openai_stub.BadRequestError = _StubBadRequestError
    sys.modules["openai"] = openai_stub

from src.rollout import rollout, MAX_TURNS  # noqa: E402


# 中文注释：通过 mock 的异步推理响应验证 rollout 的成功路径和最大轮数保护。
class TestRollout:
    """Test rollout with mocked model inference (no GPU needed)."""

# 中文注释：构造带工具调用的伪造模型 Choice。
    def _make_mock_choice(self, tool_name: str, arguments: dict):
        tc = MagicMock()
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(arguments)
        tc.id = f"call_{tool_name}"
        choice = MagicMock()
        choice.message.tool_calls = [tc]
        choice.message.content = None
        choice.finish_reason = "tool_calls"
        return choice

# 中文注释：构造终止 rollout 的纯文本 Choice。
    def _make_text_choice(self, text: str):
        choice = MagicMock()
        choice.message.tool_calls = None
        choice.message.content = text
        choice.finish_reason = "stop"
        return choice

# 中文注释：确认工具调用能被执行并产生正向奖励和正确指标。
    def test_successful_single_tool(self):
        scenario = {
            "task": "What is 3+5?",
            "expected_calls": [{"name": "calculator", "arguments": {"expression": "3+5"}}],
        }
        mock_model = MagicMock()
        mock_model.inference_api_key = "test"
        mock_model.inference_base_url = "http://test"
        mock_model.get_inference_name.return_value = "test-model"

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                resp.choices = [self._make_mock_choice("calculator", {"expression": "3+5"})]
            else:
                resp.choices = [self._make_text_choice("The answer is 8.")]
            return resp

        with patch("src.rollout.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = mock_create
            traj = asyncio.run(rollout(mock_model, scenario))

        assert traj.reward > 3.0
        assert traj.metrics["num_turns"] == 2
        assert traj.metrics["num_tools"] == 1

# 中文注释：确认模型持续调用工具时循环会在 MAX_TURNS 处停止。
    def test_max_turns_exceeded(self):
        scenario = {
            "task": "What is 3+5?",
            "expected_calls": [{"name": "calculator", "arguments": {"expression": "3+5"}}],
        }
        mock_model = MagicMock()
        mock_model.inference_api_key = "test"
        mock_model.inference_base_url = "http://test"
        mock_model.get_inference_name.return_value = "test-model"

        async def mock_create(**kwargs):
            resp = MagicMock()
            resp.choices = [self._make_mock_choice("search", {"query": "random"})]
            return resp

        with patch("src.rollout.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = mock_create
            traj = asyncio.run(rollout(mock_model, scenario))

        assert traj.reward < 0
        assert traj.metrics["num_turns"] == MAX_TURNS

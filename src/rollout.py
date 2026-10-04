"""ART rollout function for multi-step tool-use training.

Each rollout:
1. Presents the model with a task and available tools
2. Model generates tool calls, which are executed deterministically
3. Loop continues until model stops calling tools or max turns
4. Reward computed by comparing actual calls to expected calls

CRITICAL: ART's GRPO training needs per-token logprobs from the generating
policy. Without them, `backend._get_packed_tensors()` returns None and
training is skipped with the misleading "all rewards equal" message.

Two requirements:
1. Request `logprobs=True` from vLLM in every completion call
2. Store RAW Choice objects (not dict conversions) in messages_and_choices
   so ART can read choice.logprobs when building training tensors

RETRY LOGIC: vLLM 0.17.0+art1 has a concurrency bug where high parallel
request rates trigger `BadRequestError: Already borrowed` (a Rust-side
lock contention). We retry up to 3 times with exponential backoff before
giving up, which drops effective failure rate from ~70% to ~15% at the
cost of a small latency hit.
"""
# 中文注释：rollout 模块把模型推理、工具执行、轨迹记录和奖励计算串成异步循环。
import asyncio
import json
import art
from openai import AsyncOpenAI, BadRequestError
from src.tools import TOOL_SCHEMAS, execute_tool
from src.rewards import compute_reward

# Max retries for the "Already borrowed" concurrency bug in vLLM.
# Backoff schedule: 0.1s, 0.3s, 0.9s (exponential, base 3).
# 中文注释：vLLM 的并发锁竞争错误最多重试三次，使用退避等待降低瞬时失败率。
MAX_BORROWED_RETRIES = 3

MAX_TURNS = 6

# 中文注释：系统提示规定模型何时使用工具以及工具参数需要保持精确。
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use the provided tools to answer the user's question. "
    "Call tools as needed, then provide your final answer. "
    "Be precise with tool arguments."
)


# 中文注释：只为下一次 API 请求构造普通消息；训练轨迹仍保留原始 Choice 以保存 logprobs。
def _choice_to_message_dict(choice) -> dict:
    """Convert an OpenAI Choice object back to a plain message dict so it
    can be sent in the next API call. Used when feeding the growing
    conversation back to the model."""
    msg: dict = {"role": "assistant"}
    content = getattr(choice.message, "content", None)
    if content is not None:
        msg["content"] = content
    tool_calls = getattr(choice.message, "tool_calls", None)
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return msg


# 中文注释：把字典消息与 assistant Choice 混合列表转换为 OpenAI API 可接受的纯字典列表。
def _build_api_messages(messages_and_choices: list) -> list:
    """ART's messages_and_choices is a mix of dicts (system/user/tool) and
    raw Choice objects (assistant). The OpenAI API expects plain dicts, so
    we rebuild a pure dict list before each generation call."""
    out = []
    for item in messages_and_choices:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append(_choice_to_message_dict(item))
    return out


# 中文注释：一次 rollout 执行一个场景，循环生成工具调用、执行结果并最终创建 ART 轨迹。
async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    """Run a single multi-step rollout for a scenario.

    Args:
        model: ART model (provides inference endpoint)
        scenario: dict with "task" and "expected_calls"

    Returns:
        art.Trajectory with reward and metrics. On infrastructure failure
        (vLLM returning empty response, network error, etc.) returns a
        failed trajectory with reward=-3.0 so the training step doesn't
        crash on a single bad rollout.
    """
    # messages_and_choices is a mixed list: dicts for system/user/tool turns,
    # raw Choice objects for assistant turns. ART reads logprobs from the
    # Choice objects during training, so we must NOT convert them to dicts.
    messages_and_choices: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario["task"]},
    ]
    tools_called: list[str] = []
    actual_calls: list[dict] = []
    num_turns = 0

    client = AsyncOpenAI(
        api_key=model.inference_api_key,
        base_url=model.inference_base_url,
    )

    try:
        for turn in range(MAX_TURNS):
            num_turns += 1

            api_messages = _build_api_messages(messages_and_choices)

            # Higher temperature + top_p for exploration.
            # logprobs=True is REQUIRED for ART/GRPO to compute policy gradient.
            # Retry loop handles the "Already borrowed" concurrency bug in
            # vLLM 0.17.0+art1 — transient Rust-side lock contention that
            # goes away if we wait a few hundred ms and retry.
            response = None
            for retry_attempt in range(MAX_BORROWED_RETRIES):
                try:
                    response = await client.chat.completions.create(
                        model=model.get_inference_name(),
                        messages=api_messages,
                        tools=TOOL_SCHEMAS,
                        temperature=1.2,
                        top_p=0.95,
                        max_tokens=512,
                        logprobs=True,
                        top_logprobs=1,
                    )
                    break  # success
                except BadRequestError as e:
                    # Only retry the specific "Already borrowed" concurrency error;
                    # re-raise anything else immediately so the outer handler
                    # records it as a failed trajectory.
                    if "Already borrowed" not in str(e):
                        raise
                    if retry_attempt == MAX_BORROWED_RETRIES - 1:
                        raise  # out of retries, let outer handler catch it
                    await asyncio.sleep(0.1 * (3 ** retry_attempt))

            if response is None:
                # Shouldn't happen — loop either sets response or re-raises —
                # but defensive in case of future changes.
                break

            # Defensive: vLLM/ART occasionally returns responses with no choices
            # (e.g. after LoRA swap or if the engine hiccups). Treat as "model
            # said nothing" and end the rollout gracefully.
            if not response.choices:
                break
            choice = response.choices[0]

            # Store the RAW Choice object — ART reads logprobs from it during
            # training. Do NOT convert to a plain dict here; that throws away
            # the logprob data and causes _get_packed_tensors to return None.
            messages_and_choices.append(choice)

            if not choice.message.tool_calls:
                # Model emitted a text-only response — end of rollout
                break

            # Execute each tool call and append the tool result messages.
            # The assistant turn itself is already appended above as the Choice.
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tools_called.append(tool_name)
                actual_calls.append({"name": tool_name, "arguments": tool_args})
                result = execute_tool(tool_name, tool_args)

                messages_and_choices.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
    except Exception as e:
        # Any infrastructure failure during the rollout (network, vLLM, LoRA
        # routing, etc.) — return a failed trajectory with the worst reward
        # so one bad rollout doesn't crash the whole training step.
        print(f"[rollout] caught {type(e).__name__}: {e}")
        return art.Trajectory(
            messages_and_choices=messages_and_choices,
            reward=-3.0,
            metrics={
                "num_turns": float(num_turns),
                "num_tools": float(len(tools_called)),
                "failed": 1.0,
            },
        )

    expected_calls = scenario.get("expected_calls", [])
    reward = compute_reward(actual_calls, expected_calls)

    return art.Trajectory(
        messages_and_choices=messages_and_choices,
        reward=reward,
        metrics={
            "num_turns": float(num_turns),
            "num_tools": float(len(tools_called)),
            "failed": 0.0,
        },
    )

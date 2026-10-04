"""Decomposed reward function for tool-use GRPO training.

Based on ToolRL (arxiv.org/abs/2504.13958):
  R_total = R_format (0 or 1) + R_correct ([-3, 3])
  Range: [-3, 4]

Components of R_correct:
  1. Tool name Jaccard similarity
  2. Parameter key Jaccard similarity
  3. Parameter value exact match ratio
"""
# 中文注释：奖励模块解析模型消息中的 JSON 参数，并将工具轨迹转换为可比较结构。
import json


# 中文注释：Jaccard 相似度衡量两个集合的交集占并集比例，用于比较名称和参数键。
def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# 中文注释：遍历 assistant 消息，提取每个工具调用的名称和已解析参数。
def parse_tool_calls_from_messages(messages: list[dict]) -> list[dict]:
    """Extract tool calls from a conversation message list.

    Returns list of {"name": str, "arguments": dict} for each tool call.
    """
    calls = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw
            calls.append({"name": name, "arguments": args})
    return calls


# 中文注释：总奖励由“是否调用工具”的格式分和工具名、参数键、参数值匹配分组成。
def compute_reward(
    predicted_calls: list[dict],
    ground_truth_calls: list[dict],
) -> float:
    """Compute decomposed reward for a tool-use trajectory.

    Args:
        predicted_calls: list of {"name": str, "arguments": dict}
        ground_truth_calls: list of {"name": str, "arguments": dict}

    Returns:
        float in range [-3, 4]. Higher is better.
    """
    r_format = 1.0 if predicted_calls else 0.0

    if not predicted_calls:
        return r_format - 3.0  # -3.0 total

    if not ground_truth_calls:
        return r_format  # 1.0

    # Component 1: tool name Jaccard
    gt_names = {c["name"] for c in ground_truth_calls}
    pred_names = {c["name"] for c in predicted_calls}
    name_score = jaccard(gt_names, pred_names)

    # Component 2 & 3: parameter key + value matching
    matched = []
    used_pred = set()
    for g in ground_truth_calls:
        for j, p in enumerate(predicted_calls):
            if j not in used_pred and g["name"] == p["name"]:
                matched.append((g, p))
                used_pred.add(j)
                break

    param_key_score = 0.0
    param_val_score = 0.0
    match_count = max(len(matched), 1)
    total_gt_keys = sum(len(g["arguments"]) for g in ground_truth_calls) or 1

    for g_call, p_call in matched:
        g_keys = set(g_call.get("arguments", {}).keys())
        p_keys = set(p_call.get("arguments", {}).keys())
        param_key_score += jaccard(g_keys, p_keys)

        for k in g_keys & p_keys:
            if str(g_call["arguments"][k]) == str(p_call["arguments"].get(k)):
                param_val_score += 1

    # Normalize each to [0, 1], average, then scale to [-3, 3]
    raw = (name_score + param_key_score / match_count + param_val_score / total_gt_keys) / 3.0
    r_correct = 6.0 * raw - 3.0

    return r_format + r_correct

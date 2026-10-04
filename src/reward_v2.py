"""Opt-in, deterministic trajectory scoring; no ART or model dependencies.

The complete-chain checker measures the prescribed tool protocol, NOT final
answer correctness. Dynamic credit is conservatively supported for the current
weather -> conversion, country lookup -> density expression, and calculator ->
conversion routes. No symbolic algebra, tool re-execution, or judge is used.
"""

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

KNOWN_TOOLS = {"calculator", "get_weather", "search", "convert_units", "lookup_db"}


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format_score: float
    tool_score: float
    sequence_score: float
    argument_key_score: float
    argument_value_score: float
    dynamic_score: float
    completion_score: float
    baseline_score: float = -3.0
    cap_penalty: float = 0.0
    exact_sequence_match: float = 0.0
    prefix_match: float = 0.0
    tool_count_match: float = 0.0


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def normalize_value(value):
    """Numeric equivalence and case/outer-space normalization, without bool==1."""
    if isinstance(value, bool):
        return ("bool", value)
    number = _number(value)
    if number is not None:
        return number
    return value.strip().casefold() if isinstance(value, str) else value


def _expression_tree(value):
    if not isinstance(value, str) or len(value) > 1024:
        return None
    # Same numeric/operator alphabet as the existing calculator, plus whitespace.
    if not re.fullmatch(r"[0-9+*/.()\s-]+", value):
        return None
    try:
        tree = ast.parse(value.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)
    nodes = list(ast.walk(tree))
    if len(nodes) > 128 or any(not isinstance(node, allowed) for node in nodes):
        return None
    if any(isinstance(node, ast.Constant) and _number(node.value) is None for node in nodes):
        return None
    return tree


def normalize_expression(value):
    """Normalize valid numeric expression spacing/parentheses, not commutativity."""
    tree = _expression_tree(value)
    return ast.dump(tree, include_attributes=False) if tree else value


def _same_value(key, expected, predicted):
    if key == "expression":
        # Literal 15 and 15.0 are equivalent; 441+23 and 23+441 are not.
        left, right = _number(expected), _number(predicted)
        if left is not None and right is not None:
            return left == right
        return normalize_expression(expected) == normalize_expression(predicted)
    return normalize_value(expected) == normalize_value(predicted)


def _well_formed(call):
    return (
        isinstance(call, dict) and isinstance(call.get("name"), str)
        and call.get("name") in KNOWN_TOOLS and isinstance(call.get("arguments"), dict)
    )


def _decode_result(result):
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result
    return result


def _result_ok(result):
    decoded = _decode_result(result)
    if decoded is None or isinstance(decoded, bool):
        return False
    if isinstance(decoded, (int, float)):
        return _number(decoded) is not None
    if isinstance(decoded, str):
        return bool(decoded.strip()) and not decoded.strip().casefold().startswith(
            ("error:", "not found:", "no results found")
        )
    if isinstance(decoded, dict):
        return bool(decoded) and "error" not in decoded
    if isinstance(decoded, list):
        return bool(decoded) and all(_result_ok(item) for item in decoded) and not any(
            isinstance(item, dict) and str(item.get("text", "")).casefold().startswith("no results found")
            for item in decoded
        )
    return False


def _trace_matches(entry, call):
    return (
        isinstance(entry, dict) and "result" in entry
        and entry.get("name") == call.get("name")
        and entry.get("arguments") == call.get("arguments")
    )


def _previous_result(predicted_calls, observations, index):
    """Never use a future/current result; full traces must bind to the prior call.

    Result-only observations are accepted for standalone reward calculations.
    The stricter execution checker below requires full, aligned trace entries.
    """
    if index < 1 or observations is None or index - 1 >= len(observations):
        return None
    entry = observations[index - 1]
    if isinstance(entry, dict) and "result" in entry:
        if not _trace_matches(entry, predicted_calls[index - 1]):
            return None
        current = observations[index] if index < len(observations) else None
        if "turn" in entry or (isinstance(current, dict) and "turn" in current):
            # Same assistant Choice can propose several tools before seeing ANY
            # result. Sequential execution alone is not observation reuse.
            if not isinstance(current, dict) or not _trace_matches(current, predicted_calls[index]):
                return None
            previous_turn, current_turn = entry.get("turn"), current.get("turn")
            if (not isinstance(previous_turn, int) or isinstance(previous_turn, bool)
                    or not isinstance(current_turn, int) or isinstance(current_turn, bool)):
                return None
            if previous_turn < 1 or current_turn <= previous_turn:
                return None
        entry = entry["result"]
    return _decode_result(entry) if _result_ok(entry) else None


def _literal(node):
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _dynamic_matches(key, value, previous_name, result, arguments):
    if key == "value":
        number = _number(value)
        if number is None:
            return False
        if previous_name == "get_weather" and isinstance(result, dict):
            # Humidity must not masquerade as temperature; units must agree.
            unit = result.get("units", "celsius")
            return (
                number == _number(result.get("temperature"))
                and normalize_value(arguments.get("from_unit")) == normalize_value(unit)
            )
        if previous_name in {"calculator", "convert_units"}:
            return number == _number(result)
        return False
    if key == "expression" and previous_name == "lookup_db" and isinstance(result, dict):
        # Current dynamic calculator tasks are population / area. Require BOTH
        # observed operands in the prescribed order; no arbitrary numeric hit.
        tree = _expression_tree(value)
        population, area = _number(result.get("population")), _number(result.get("area_km2"))
        return bool(
            tree and isinstance(tree.body, ast.BinOp) and isinstance(tree.body.op, ast.Div)
            and population is not None and area is not None and area != 0
            and _literal(tree.body.left) == population and _literal(tree.body.right) == area
        )
    return False


def _argument_matches(key, expected, predicted, predicted_calls, expected_calls, observations, index):
    if expected != "__dynamic__":
        return _same_value(key, expected, predicted)
    if index < 1 or any(
        predicted_calls[i].get("name") != expected_calls[i].get("name") for i in range(index)
    ):
        return False
    result = _previous_result(predicted_calls, observations, index)
    return _dynamic_matches(
        key, predicted, predicted_calls[index - 1].get("name"), result,
        predicted_calls[index].get("arguments", {}),
    )


def _arguments_complete(predicted, expected, predicted_calls, expected_calls, observations, index):
    p_args, g_args = predicted["arguments"], expected["arguments"]
    if not all(
        key in p_args and _argument_matches(
            key, value, p_args[key], predicted_calls, expected_calls, observations, index,
        )
        for key, value in g_args.items()
    ):
        return False
    extras = set(p_args) - set(g_args)
    if predicted["name"] == "get_weather" and "units" not in g_args:
        # An explicitly specified default is benign; Fahrenheit is not.
        if "units" in p_args and normalize_value(p_args["units"]) != "celsius":
            return False
        extras.discard("units")
    return not extras


def _calls_complete(predicted_calls, expected_calls, observations):
    return bool(expected_calls) and len(predicted_calls) == len(expected_calls) and all(
        _well_formed(p) and _well_formed(g) and p["name"] == g["name"]
        and _arguments_complete(p, g, predicted_calls, expected_calls, observations, i)
        for i, (p, g) in enumerate(zip(predicted_calls, expected_calls, strict=True))
    )


def is_complete_tool_chain(predicted_calls, expected_calls, *, observations=None) -> bool:
    """Frozen complete-tool-chain-v1 predicate, independent of training reward.

    Exact call count/order, normalized required arguments, supported dynamic
    dependencies, and aligned non-error execution results are required. Missing
    trace is NOT evidence of success. This does not check the final answer.
    """
    if observations is None or len(observations) != len(predicted_calls):
        return False
    return _calls_complete(predicted_calls, expected_calls, observations) and all(
        _trace_matches(entry, call) and _result_ok(entry["result"])
        for entry, call in zip(observations, predicted_calls, strict=True)
    )


def compute_reward_v2(predicted_calls, ground_truth_calls, *, observations=None) -> RewardBreakdown:
    """Weighted contributions sum to [-3, 4], with a <=2 cap for incomplete calls.

    Format=1; tool multiset overlap=1; sequence=1.5; argument keys=1;
    static+dynamic values=1.5; completion=1; baseline=-3. Missing/reordered
    calls get partial credit, not an all-or-nothing penalty. The cap preserves
    the legacy >2 threshold against partial-chain false positives.
    """
    if not predicted_calls:
        return RewardBreakdown(-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    well_formed = all(_well_formed(call) for call in predicted_calls)
    if not well_formed or not all(_well_formed(call) for call in ground_truth_calls):
        return RewardBreakdown(-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if not ground_truth_calls:
        return RewardBreakdown(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, baseline_score=0.0)

    p_names = [call["name"] for call in predicted_calls]
    g_names = [call["name"] for call in ground_truth_calls]
    denominator = max(len(p_names), len(g_names))
    tool_score = sum((Counter(p_names) & Counter(g_names)).values()) / denominator
    prefix = 0
    for p_name, g_name in zip(p_names, g_names, strict=False):
        if p_name != g_name:
            break
        prefix += 1
    prefix_match = prefix / denominator
    exact = float(p_names == g_names)
    count_match = min(len(p_names), len(g_names)) / denominator
    sequence_score = 1.5 * (0.5 * prefix_match + 0.25 * exact + 0.25 * count_match)
    key_total = static_matches = dynamic_matches = 0.0
    value_total = sum(len(call["arguments"]) for call in ground_truth_calls) or 1
    for i, (predicted, expected) in enumerate(zip(predicted_calls, ground_truth_calls, strict=False)):
        if predicted["name"] != expected["name"]:
            continue
        p_args, g_args = predicted["arguments"], expected["arguments"]
        union = set(p_args) | set(g_args)
        key_total += len(set(p_args) & set(g_args)) / len(union) if union else 1.0
        for key, value in g_args.items():
            if key in p_args and _argument_matches(
                key, value, p_args[key], predicted_calls, ground_truth_calls, observations, i,
            ):
                if value == "__dynamic__":
                    dynamic_matches += 1
                else:
                    static_matches += 1
    key_score = key_total / denominator
    value_score = 1.5 * static_matches / value_total
    dynamic_score = 1.5 * dynamic_matches / value_total
    complete = _calls_complete(predicted_calls, ground_truth_calls, observations)
    if observations is not None:
        # When execution evidence is supplied, explicit errors/misaligned or
        # missing entries cannot receive completion credit or >2 reward.
        complete = complete and is_complete_tool_chain(
            predicted_calls, ground_truth_calls, observations=observations,
        )
    completion_score = float(complete)
    raw = -3.0 + 1.0 + tool_score + sequence_score + key_score + value_score + dynamic_score + completion_score
    cap_penalty = min(0.0, 2.0 - raw) if not complete else 0.0
    return RewardBreakdown(
        raw + cap_penalty, 1.0, tool_score, sequence_score, key_score,
        value_score, dynamic_score, completion_score, cap_penalty=cap_penalty,
        exact_sequence_match=exact, prefix_match=prefix_match, tool_count_match=count_match,
    )


def score_final_answer(*args, **kwargs) -> float:
    """Reserved interface only: no judge and no final-answer reward this phase."""
    return 0.0

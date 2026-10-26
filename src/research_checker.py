"""Independent answer result + separate replay/protocol/dependency diagnostics."""

import ast
import json

from .research_answers import check_answer, final_content, number
from .research_environment import ResearchEnvironment

TRACE_VERSION = "research-trace-v1"
MAX_TRACE_STEPS = 64


def expression_signature(expression: str) -> tuple:
    """Normalize parentheses/numeric spellings and commutative +/*, not arbitrary algebra."""
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("invalid expression length")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant):
            return ("number", number(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            child = visit(node.operand)
            if child[0] == "number":
                return ("number", -child[1] if isinstance(node.op, ast.USub) else child[1])
            return (type(node.op).__name__, child)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            children = [visit(node.left), visit(node.right)]
            if isinstance(node.op, (ast.Add, ast.Mult)):
                children.sort(key=repr)
            return (type(node.op).__name__, *children)
        raise ValueError("unsupported arithmetic")

    return visit(tree.body)


def call_signature(name: str, arguments: dict) -> tuple:
    args = dict(arguments)
    if name == "get_weather":
        args.setdefault("units", "celsius")
    if name == "calculator" and set(args) == {"expression"}:
        return name, expression_signature(args["expression"])
    # JSON numbers compare numerically but bools never match numbers (replay rejects them).
    return name, tuple(sorted(args.items()))


def protocol_plan(record: dict) -> tuple[list[dict], list[tuple[int, int]]]:
    """Construct supported route from actual observations, never saved trace/gold/edges.

    The trace checker shares the executor with rollouts, unlike the answer oracle.
    This defines protocol evidence only; equivalent unregistered routes can still
    receive answer credit. Independent comparisons may be queried in either order.
    """
    env = ResearchEnvironment(record["environment_snapshot"])
    recipe, p = record["recipe_id"], record["parameters"]
    steps, dependencies = [], []

    def call(name, arguments, producer=None):
        observation = env.execute(name, arguments)
        if observation.startswith("Error:"):
            raise ValueError(f"invalid protocol fixture: {observation}")
        if producer is not None:
            dependencies.append((producer, len(steps)))
        steps.append({"name": name, "arguments": arguments})
        try:
            return json.loads(observation)
        except json.JSONDecodeError as exc:
            raise ValueError("executor returned non-JSON observation") from exc

    def conversion(value, source, target, producer=None):
        return call("convert_units", {"value": value, "from_unit": source, "to_unit": target}, producer)

    if recipe == "simple_math":
        call("calculator", {"expression": f"{p['a']}{p['op']}{p['b']}"})
    elif recipe == "simple_convert":
        conversion(p["value"], p["from_unit"], p["to_unit"])
    elif recipe in {"simple_weather", "weather_convert", "weather_offset_convert", "weather_compare"}:
        value = call("get_weather", {"city": p["entity"], "units": "celsius"})["temperature"]
        if recipe == "weather_compare":
            call("get_weather", {"city": p["entity_b"], "units": "celsius"})
        elif recipe != "simple_weather":
            if recipe == "weather_offset_convert":
                value = call("calculator", {"expression": f"{value}+{p['offset']}"}, 0)
            conversion(value, "celsius", "fahrenheit", len(steps) - 1)
    elif recipe in {"simple_search", "age_calculation", "search_and_convert", "measurement_scale", "distance_offset_convert"}:
        field = "birth_year" if recipe == "age_calculation" else "distance_km"
        value = call("search", {"query": f"{p['entity']} {field}"})[0]["value"]
        if recipe == "age_calculation":
            call("calculator", {"expression": f"{p['year']}-{value}"}, 0)
        elif recipe in {"measurement_scale", "distance_offset_convert"}:
            op, key = ("*", "scale") if recipe == "measurement_scale" else ("+", "offset")
            value = call("calculator", {"expression": f"{value}{op}{p[key]}"}, 0)
        if recipe in {"search_and_convert", "distance_offset_convert"}:
            conversion(value, "km", "miles", len(steps) - 1)
    elif recipe in {"simple_lookup", "gdp_scale", "population_math", "country_density_convert", "country_compare"}:
        fact = call("lookup_db", {"table": "countries", "key": p["entity"]})
        if recipe == "country_compare":
            call("lookup_db", {"table": "countries", "key": p["entity_b"]})
        elif recipe == "gdp_scale":
            call("calculator", {"expression": f"{fact['gdp_billion_usd']}*{p['scale']}"}, 0)
        elif recipe in {"population_math", "country_density_convert"}:
            density = call("calculator", {"expression": f"{fact['population']}/{fact['area_km2']}"}, 0)
            if recipe == "country_density_convert":
                conversion(density, "people_per_km2", "people_per_mile2", 1)
    else:
        raise ValueError("unregistered protocol recipe")
    return steps, dependencies


def check_trace(record: dict, trace: list[dict] | None, *, termination: str = "finished") -> dict:
    """Flat trace events: turn/name/arguments/observation; same-turn deps fail.

    Tool results must exactly equal replay in the immutable world. Turn is the
    producing assistant turn (not a fabricated turn per function in a batch).
    Call IDs/message provenance must be supplied by a future model-run adapter.
    """
    plan, edges = protocol_plan(record)
    result = {"version": TRACE_VERSION, "execution_valid": False, "strict_complete": False,
              "semantic_valid": False, "dynamic_status": "invalid" if edges else "not_applicable",
              "matched_steps": [], "errors": []}
    errors = result["errors"]
    if termination not in {"finished", "truncated", "exception", "infra_failure", "missing_output"}:
        raise ValueError("unknown termination")
    if termination != "finished":
        errors.append(termination)
    if not isinstance(trace, list) or len(trace) > MAX_TRACE_STEPS:
        errors.append("invalid_trace")
        return result
    env, signatures = ResearchEnvironment(record["environment_snapshot"]), []
    previous_turn = -1
    for step in trace:
        try:
            if (not isinstance(step, dict) or set(step) != {"turn", "name", "arguments", "observation"}
                    or type(step["turn"]) is not int or step["turn"] < previous_turn or step["turn"] < 0
                    or not isinstance(step["name"], str) or not isinstance(step["arguments"], dict)
                    or not isinstance(step["observation"], str)):
                raise ValueError("invalid trace event")
            previous_turn = step["turn"]
            replay = env.execute(step["name"], step["arguments"])
            if replay.startswith("Error:"):
                errors.append("tool_execution_error")
            elif replay != step["observation"]:
                errors.append("fabricated_observation")
            signatures.append(call_signature(step["name"], step["arguments"]))
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
            errors.append("invalid_trace")
            return result
    result["execution_valid"] = not any(e in errors for e in {"tool_execution_error", "fabricated_observation", "invalid_trace"})
    if not trace:
        errors.append("no_tool")
    expected = [call_signature(s["name"], s["arguments"]) for s in plan]
    candidates = [[i for i, signature in enumerate(signatures) if signature == target] for target in expected]

    def match(position, selected, enforce_turns):
        if position == len(plan):
            return selected
        for index in candidates[position]:
            if index in selected:
                continue
            valid = True
            for source, target in edges:
                if target == position and (index <= selected[source] or
                        (enforce_turns and trace[index]["turn"] <= trace[selected[source]]["turn"])):
                    valid = False
            if valid:
                found = match(position + 1, [*selected, index], enforce_turns)
                if found is not None:
                    return found
        return None

    mapping = match(0, [], True)
    if mapping is None:
        if edges and match(0, [], False) is not None:
            errors.append("same_turn_dependency")
        else:
            errors.append("missing_step" if len(trace) < len(plan) else "wrong_arguments_or_route")
    else:
        result["matched_steps"] = mapping
        if edges:
            result["dynamic_status"] = "valid" if result["execution_valid"] and termination == "finished" else "invalid"
        result["semantic_valid"] = result["execution_valid"] and termination == "finished"
        result["strict_complete"] = result["semantic_valid"] and len(trace) == len(plan)
        if len(trace) != len(plan):
            errors.append("extra_calls")
    # Registered one-call alternative; no reference edge claimed as causally validated.
    alternate = call_signature("get_weather", {"city": record["parameters"].get("entity"), "units": "fahrenheit"})
    if record["recipe_id"] == "weather_convert" and mapping is None and alternate in signatures and result["execution_valid"] and termination == "finished":
        result["semantic_valid"] = True
        result["dynamic_status"] = "alternative_bypass"
        errors[:] = [e for e in errors if e not in {"missing_step", "wrong_arguments_or_route"}]
        errors.append("legal_alternative")
    result["errors"] = sorted(set(errors))
    return result


def check_attempt(record: dict, content: str, trace: list[dict], *, termination: str = "finished") -> dict:
    """Answers never gated by route/reward. Unfinished attempts cannot claim success."""
    answer = check_answer(record, content)
    if termination != "finished":
        answer["correct"] = False
        answer["reason"] = termination
    return {"answer": answer, "trace": check_trace(record, trace, termination=termination)}


def check_messages(record: dict, messages: list[dict], trace: list[dict], *, termination: str = "finished") -> dict:
    """Normalized chat adapter: only terminal non-tool assistant can supply an answer."""
    try:
        content = final_content(messages)
    except ValueError:
        return check_attempt(record, "", trace, termination="missing_output" if termination == "finished" else termination)
    return check_attempt(record, content, trace, termination=termination)

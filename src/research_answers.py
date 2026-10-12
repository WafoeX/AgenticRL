"""R2 answer oracle: stdlib only; no tool/reward/generator/reference imports."""

import json
import math
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

ANSWER_VERSION = "research-answer-v1"
MAX_ANSWER_BYTES = 16384
UNITS = frozenset({
    "dimensionless", "years", "celsius", "fahrenheit", "km", "miles", "kg", "lbs",
    "meters", "feet", "people_per_km2", "people_per_mile2", "billion_usd", "entity", "condition",
})
INTEGER_RECIPES = frozenset({"simple_math", "age_calculation", "simple_lookup", "gdp_scale"})


def number(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("expected finite number, not boolean/string")
    result = Decimal(str(value))
    try:
        finite = result.is_finite() and math.isfinite(float(result))
    except (ValueError, OverflowError):
        finite = False
    if not finite:
        raise ValueError("non-finite/out-of-range number")
    return result


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_answer(content: str) -> dict:
    """No extraction/repair/aliases. One object, exactly value/unit, finite scalar."""
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_ANSWER_BYTES:
        raise ValueError("answer must be a bounded UTF-8 string")
    try:
        answer = json.loads(content, parse_float=Decimal, parse_int=Decimal,
                            parse_constant=reject_constant, object_pairs_hook=unique_object)
        if not isinstance(answer, dict) or set(answer) != {"value", "unit"}:
            raise ValueError("expected exactly value and unit")
        if not isinstance(answer["unit"], str) or answer["unit"] not in UNITS:
            raise ValueError("unknown unit")
        if answer["unit"] in {"entity", "condition"}:
            if not isinstance(answer["value"], str) or not answer["value"]:
                raise ValueError("expected nonempty enum string")
        else:
            number(answer["value"])
        return answer
    except (json.JSONDecodeError, ArithmeticError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid answer JSON") from exc


def independent_conversion(value: object, source: str, target: str) -> Decimal:
    """Fixed SI definitions + six-place half-even; NOT the environment converter."""
    with localcontext() as ctx:
        ctx.prec = 50
        v = number(value)
        definitions = {"km": ("length", Decimal(1000)), "miles": ("length", Decimal("1609.344")),
                       "meters": ("short_length", Decimal(1)), "feet": ("short_length", Decimal("0.3048")),
                       "kg": ("mass", Decimal(1)), "lbs": ("mass", Decimal("0.45359237")),
                       "people_per_km2": ("density", Decimal(1)),
                       "people_per_mile2": ("density", Decimal(1) / Decimal("1.609344") ** 2)}
        if (source, target) == ("celsius", "fahrenheit"):
            result = v * Decimal(9) / 5 + 32
        elif (source, target) == ("fahrenheit", "celsius"):
            result = (v - 32) * 5 / 9
        elif source != target and source in definitions and target in definitions:
            a, b = definitions[source], definitions[target]
            if a[0] != b[0]:
                raise ValueError("incompatible units")
            result = v * a[1] / b[1]
        else:
            raise ValueError("unsupported conversion")
        return result.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)


def oracle_answer(record: dict) -> dict:
    """Only recipe, public parameters and raw environment facts; ignore saved golds."""
    p, world, recipe = record["parameters"], record["environment_snapshot"], record["recipe_id"]
    with localcontext() as ctx:
        ctx.prec = 50
        if recipe == "simple_math":
            a, b = number(p["a"]), number(p["b"])
            operations = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}
            value, unit = operations[p["op"]](), "dimensionless"
        elif recipe == "simple_convert":
            value, unit = independent_conversion(p["value"], p["from_unit"], p["to_unit"]), p["to_unit"]
        elif recipe == "age_calculation":
            value = number(p["year"]) - number(world["search"][p["entity"]]["birth_year"]["value"])
            unit = "years"
        elif recipe in {"weather_compare", "country_compare"}:
            table, field = ("weather", "temperature") if recipe == "weather_compare" else ("countries", "population")
            a, b = (number(world[table][p[key]][field]) for key in ("entity", "entity_b"))
            return {"value": "tie" if a == b else p["entity"] if a > b else p["entity_b"], "unit": "entity"}
        elif recipe in {"simple_weather", "weather_convert", "weather_offset_convert"}:
            value = number(world["weather"][p["entity"]]["temperature"])
            if recipe == "weather_offset_convert":
                value += number(p["offset"])
            unit = "celsius" if recipe == "simple_weather" else "fahrenheit"
            if unit == "fahrenheit":
                value = independent_conversion(value, "celsius", unit)
        elif recipe in {"simple_search", "search_and_convert", "measurement_scale", "distance_offset_convert"}:
            value = number(world["search"][p["entity"]]["distance_km"]["value"])
            if recipe == "measurement_scale":
                value *= number(p["scale"])
            elif recipe == "distance_offset_convert":
                value += number(p["offset"])
            unit = "miles" if recipe in {"search_and_convert", "distance_offset_convert"} else "km"
            if unit == "miles":
                value = independent_conversion(value, "km", unit)
        elif recipe in {"simple_lookup", "gdp_scale", "population_math", "country_density_convert"}:
            fact = world["countries"][p["entity"]]
            if recipe in {"simple_lookup", "gdp_scale"}:
                value = number(fact["gdp_billion_usd"])
                if recipe == "gdp_scale":
                    value *= number(p["scale"])
                unit = "billion_usd"
            else:
                area = number(fact["area_km2"])
                if area <= 0:
                    raise ValueError("nonpositive area")
                value = number(fact["population"]) / area
                unit = "people_per_km2" if recipe == "population_math" else "people_per_mile2"
                if unit == "people_per_mile2":
                    value = independent_conversion(value, "people_per_km2", unit)
        else:
            raise ValueError(f"unregistered recipe: {recipe}")
        number(value)
        integral = recipe in INTEGER_RECIPES and value == value.to_integral_value()
        return {"value": int(value) if integral else float(value), "unit": unit}


def answer_specification(record: dict) -> dict:
    """Frozen checker policy, not model-controlled or mutable record tolerances."""
    gold = oracle_answer(record)
    exact = record["recipe_id"] in INTEGER_RECIPES or gold["unit"] == "entity"
    return {"type": "enum" if gold["unit"] == "entity" else "number", "unit": gold["unit"],
            "abs_tol": 0 if exact else 1e-6, "rel_tol": 0 if exact else 1e-9}


def answers_equal(prediction: dict, gold: dict, specification: dict) -> bool:
    if prediction["unit"] != gold["unit"]:
        return False
    if specification["type"] == "enum":
        return prediction["value"] == gold["value"]
    with localcontext() as ctx:
        ctx.prec = 50
        a, b = number(prediction["value"]), number(gold["value"])
        return abs(a - b) <= max(number(specification["abs_tol"]), number(specification["rel_tol"]) * abs(b))


def final_content(messages: list[dict]) -> str:
    """Terminal assistant message only: never fall back to an earlier correct answer."""
    if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("missing terminal assistant answer")
    content = messages[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("missing terminal answer content")
    return content


def check_answer(record: dict, content: str) -> dict:
    gold, spec = oracle_answer(record), answer_specification(record)
    try:
        parsed = parse_answer(content)
        correct = answers_equal(parsed, gold, spec)
        value = parsed["value"]
        if isinstance(value, Decimal):
            value = int(value) if value == value.to_integral_value() else float(value)
        return {"version": ANSWER_VERSION, "parsed": {"value": value, "unit": parsed["unit"]},
                "correct": correct, "reason": "correct" if correct else "unit_error" if parsed["unit"] != gold["unit"] else "value_error"}
    except (ValueError, ArithmeticError, RecursionError, UnicodeError) as exc:
        return {"version": ANSWER_VERSION, "parsed": None, "correct": False, "reason": "parse_error", "detail": str(exc)}

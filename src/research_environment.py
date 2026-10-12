"""Isolated Research V1 tools; no legacy state, network, eval, or global RNG."""

import ast
import json
import math
import operator
from dataclasses import dataclass

ENVIRONMENT_VERSION = "synthetic-tools-v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def numeric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected finite number (not boolean)")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("expected finite number") from exc
    if not math.isfinite(result):
        raise ValueError("expected finite number")
    return result


def calculate(expression: str) -> float:
    """Bounded arithmetic AST: no names/calls/powers, finite results only."""
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("expression must be a string of at most 256 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant):
            return numeric(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return numeric(visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1))
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return numeric(binary[type(node.op)](visit(node.left), visit(node.right)))
        raise ValueError("only numeric +-*/ arithmetic is supported")

    return visit(tree.body)


CONVERSION_FACTORS = {
    ("km", "miles"): 1 / 1.609344, ("miles", "km"): 1.609344,
    ("kg", "lbs"): 1 / 0.45359237, ("lbs", "kg"): 0.45359237,
    ("meters", "feet"): 1 / 0.3048, ("feet", "meters"): 0.3048,
    ("people_per_km2", "people_per_mile2"): 1.609344 ** 2,
    ("people_per_mile2", "people_per_km2"): 1 / 1.609344 ** 2,
}
CONVERSION_PAIRS = list(CONVERSION_FACTORS) + [("celsius", "fahrenheit"), ("fahrenheit", "celsius")]


def convert(value: object, source: str, target: str) -> float:
    value = numeric(value)
    if (source, target) == ("celsius", "fahrenheit"):
        result = value * 1.8 + 32
    elif (source, target) == ("fahrenheit", "celsius"):
        result = (value - 32) / 1.8
    elif (source, target) in CONVERSION_FACTORS:
        result = value * CONVERSION_FACTORS[source, target]
    else:
        raise ValueError("unsupported conversion")
    return round(numeric(result), 6)


# Search intentionally has an exact entity/field contract, not fuzzy legacy matching.
_TOOL_PARAMETERS = {
    "calculator": ({"expression": {"type": "string"}}, ["expression"]),
    "get_weather": ({"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "search": ({"query": {"type": "string", "description": "Exact query: <entity> <field>; fields: birth_year, distance_km"}}, ["query"]),
    "convert_units": ({"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, ["value", "from_unit", "to_unit"]),
    "lookup_db": ({"table": {"type": "string", "enum": ["countries"]}, "key": {"type": "string"}}, ["table", "key"]),
}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": name, "description": description,
     "parameters": {"type": "object", "properties": _TOOL_PARAMETERS[name][0],
                    "required": _TOOL_PARAMETERS[name][1], "additionalProperties": False}}}
    for name, description in {
        "calculator": "Bounded numeric arithmetic with + - * / and parentheses.",
        "get_weather": "Read synthetic city temperature, condition, humidity; default Celsius.",
        "search": "Read an exact synthetic entity/field fact; returns [{entity, field, value, unit}].",
        "convert_units": "Convert temperature, km/miles, kg/lbs, meters/feet, people_per_km2/people_per_mile2; round to 6 decimals.",
        "lookup_db": "Read a synthetic countries record: population, area_km2, gdp_billion_usd.",
    }.items()
]


@dataclass(frozen=True, init=False)
class ResearchEnvironment:
    """Immutable serialized snapshot; accessor and every result are defensive copies."""

    _snapshot_json: str

    def __init__(self, snapshot: dict):
        if not isinstance(snapshot, dict) or set(snapshot) != {"version", "weather", "search", "countries"} or snapshot["version"] != ENVIRONMENT_VERSION:
            raise ValueError("invalid environment snapshot/version")
        if not all(isinstance(snapshot[k], dict) for k in ("weather", "search", "countries")):
            raise ValueError("invalid environment tables")
        for table in ("weather", "search", "countries"):
            for entity, record in snapshot[table].items():
                if not isinstance(entity, str) or not entity or not isinstance(record, dict):
                    raise ValueError("invalid entity/record")
                if table == "weather":
                    if set(record) != {"temperature", "condition", "humidity"} or record["condition"] not in {"sunny", "cloudy", "rainy"}:
                        raise ValueError("invalid weather record")
                    numeric(record["temperature"])
                    if not 0 <= numeric(record["humidity"]) <= 100:
                        raise ValueError("humidity out of range")
                elif table == "countries":
                    if set(record) != {"population", "area_km2", "gdp_billion_usd"}:
                        raise ValueError("invalid country record")
                    if type(record["population"]) is not int or record["population"] < 0 or numeric(record["area_km2"]) <= 0 or numeric(record["gdp_billion_usd"]) < 0:
                        raise ValueError("invalid country values")
                else:
                    if set(record) != {"birth_year", "distance_km"}:
                        raise ValueError("invalid search record")
                    for field, unit in (("birth_year", "years"), ("distance_km", "km")):
                        fact = record[field]
                        if not isinstance(fact, dict) or set(fact) != {"value", "unit"} or fact["unit"] != unit or numeric(fact["value"]) < 0:
                            raise ValueError("invalid search fact")
                    if type(record["birth_year"]["value"]) is not int:
                        raise ValueError("birth_year must be an integer")
        object.__setattr__(self, "_snapshot_json", canonical_json(snapshot))

    @property
    def snapshot(self) -> dict:
        try:
            return json.loads(self._snapshot_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("corrupt internal environment serialization") from exc

    def execute(self, name: str, arguments: dict) -> str:
        try:
            if name not in _TOOL_PARAMETERS or not isinstance(arguments, dict):
                raise ValueError("unknown tool or invalid arguments")
            properties, required = _TOOL_PARAMETERS[name]
            if set(arguments) - set(properties) or not set(required) <= set(arguments):
                raise ValueError("missing or extra arguments")
            for key, value in arguments.items():
                if properties[key]["type"] == "number":
                    numeric(value)
                elif not isinstance(value, str):
                    raise ValueError("expected string argument")
                if "enum" in properties[key] and value not in properties[key]["enum"]:
                    raise ValueError("unsupported enum argument")
            world = self.snapshot
            if name == "calculator":
                return canonical_json(calculate(arguments["expression"]))
            if name == "convert_units":
                return canonical_json(convert(arguments["value"], arguments["from_unit"], arguments["to_unit"]))
            if name == "get_weather":
                record = world["weather"][arguments["city"]].copy()
                units = arguments.get("units", "celsius")
                if units == "fahrenheit":
                    record["temperature"] = convert(record["temperature"], "celsius", units)
                return canonical_json({**record, "units": units})
            if name == "search":
                entity, field = arguments["query"].rsplit(" ", 1)
                fact = world["search"][entity][field]
                return canonical_json([{"entity": entity, "field": field, **fact}])
            return canonical_json(world["countries"][arguments["key"]])
        except (ValueError, KeyError, TypeError, SyntaxError, ArithmeticError, RecursionError) as exc:
            return f"Error: {exc}"

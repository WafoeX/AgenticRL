"""Research record contracts and canonical identities (stdlib only)."""

import hashlib
import json
import re
import unicodedata

from .research_environment import (
    ENVIRONMENT_VERSION,
    ResearchEnvironment,
    canonical_json,
    numeric,
)

SCHEMA_VERSION = "research-record-v1"
SYSTEM_PROMPT = (
    "This benchmark uses fictional entities and fixed synthetic facts, not real-world facts. "
    "Use the provided tools to read this scenario's facts. Search queries have the form '<entity> <field>'. "
    "The final response must be exactly one JSON object with keys value and unit. "
    "value is a finite number or an entity/condition string; equal comparisons use tie. "
    "Use the requested unit, with these unit labels: dimensionless, years, celsius, fahrenheit, "
    "km, miles, kg, lbs, meters, feet, people_per_km2, people_per_mile2, billion_usd, entity, condition."
)
UNITS = {
    "dimensionless", "years", "celsius", "fahrenheit", "km", "miles", "kg", "lbs", "meters", "feet",
    "people_per_km2", "people_per_mile2", "billion_usd", "entity", "condition",
}
RECORD_TYPES = {
    "schema_version": str, "dataset_version": str, "sample_id": str, "family": str, "recipe_id": str,
    "task": str, "wording_id": str, "semantic_group_id": str, "semantic_signature": str,
    "value_signature": str, "trajectory_signature": str, "split": str, "generalization": str,
    "parameters": dict, "parameter_domain": str, "candidate_index": int, "environment_seed": int,
    "environment_version": str, "environment_snapshot": dict, "environment_hash": str,
    "reference_depth": int, "dependency_graph": list, "expected_calls": list, "reference_trace": list,
    "answer_spec": dict, "oracle_answer": dict, "reference_answer": dict, "semantic_payload": dict,
    "value_payload": dict, "trajectory_payload": list, "dynamic_dependency": bool,
}


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def model_messages(record: dict) -> list[dict]:
    """Only public task + fixed protocol. Never serialize an entire record to the model."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": record["task"]}]


def answers_agree(answer: dict, gold: dict, spec: dict) -> bool:
    """Internal reference verification, NOT the final model-answer parser/checker (R2)."""
    if answer["unit"] != gold["unit"]:
        return False
    if spec["type"] == "enum":
        return answer["value"] == gold["value"]
    return abs(numeric(answer["value"]) - numeric(gold["value"])) <= max(
        spec["abs_tol"], spec["rel_tol"] * abs(numeric(gold["value"]))
    )


def validate_record(record: dict, *, replay: bool = True) -> None:
    if not isinstance(record, dict) or set(record) != set(RECORD_TYPES):
        raise ValueError("record fields differ from schema")
    for key, expected in RECORD_TYPES.items():
        if type(record[key]) is not expected:
            raise ValueError(f"wrong type for {key}")
    # Reject non-finite JSON anywhere, including hidden metadata.
    canonical_json(record)
    if record["schema_version"] != SCHEMA_VERSION or record["environment_version"] != ENVIRONMENT_VERSION:
        raise ValueError("unknown schema/environment version")
    if record["split"] not in {"train", "validation", "test"} or record["generalization"] not in {"train", "iid", "parameter", "wording", "composition"}:
        raise ValueError("invalid split/generalization")
    if (record["split"] == "train") != (record["generalization"] == "train"):
        raise ValueError("invalid training generalization")
    if record["candidate_index"] < 0 or record["environment_seed"] < 0 or not record["task"].strip():
        raise ValueError("invalid index/seed/task")
    for field, payload in [("environment_hash", "environment_snapshot"), ("semantic_signature", "semantic_payload"),
                           ("value_signature", "value_payload"), ("trajectory_signature", "trajectory_payload")]:
        if record[field] != digest(record[payload]):
            raise ValueError(f"invalid {field}")
    if record["semantic_group_id"] != record["semantic_signature"] or record["sample_id"] != "rv1-" + record["semantic_signature"]:
        raise ValueError("invalid group/sample identity")
    depth = record["reference_depth"]
    trace, calls, edges = record["reference_trace"], record["expected_calls"], record["dependency_graph"]
    if depth not in (1, 2, 3) or len(trace) != depth or len(calls) != depth:
        raise ValueError("invalid reference depth")
    if record["dynamic_dependency"] != bool(edges):
        raise ValueError("dynamic flag differs from graph")
    if "__dynamic__" in canonical_json(calls):
        raise ValueError("unresolved dynamic parameter")
    env = ResearchEnvironment(record["environment_snapshot"])
    for i, (call, step) in enumerate(zip(calls, trace, strict=True)):
        if set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("invalid call")
        if set(step) != {"turn", "name", "arguments", "observation"} or step["turn"] != i:
            raise ValueError("invalid trace step")
        if call != {"name": step["name"], "arguments": step["arguments"]} or not isinstance(step["observation"], str):
            raise ValueError("trace/call mismatch")
        if replay and (step["observation"].startswith("Error:") or env.execute(call["name"], call["arguments"]) != step["observation"]):
            raise ValueError("reference execution mismatch")
    for edge in edges:
        if set(edge) != {"source", "target", "source_fields", "target_argument", "transform"}:
            raise ValueError("invalid dependency edge")
        if type(edge["source"]) is not int or type(edge["target"]) is not int or not 0 <= edge["source"] < edge["target"] < depth:
            raise ValueError("dependency order invalid")
        if not isinstance(edge["source_fields"], list) or not edge["source_fields"] or not all(isinstance(v, str) for v in edge["source_fields"]):
            raise ValueError("invalid dependency fields")
        if edge["target_argument"] not in calls[edge["target"]]["arguments"] or not isinstance(edge["transform"], str):
            raise ValueError("invalid dependency target")
    spec = record["answer_spec"]
    if set(spec) != {"type", "unit", "abs_tol", "rel_tol"} or spec["type"] not in {"number", "enum"} or spec["unit"] not in UNITS:
        raise ValueError("invalid answer specification")
    if numeric(spec["abs_tol"]) < 0 or numeric(spec["rel_tol"]) < 0:
        raise ValueError("invalid tolerance")
    for answer in (record["oracle_answer"], record["reference_answer"]):
        if set(answer) != {"value", "unit"} or answer["unit"] != spec["unit"]:
            raise ValueError("invalid answer fields/unit")
        if spec["type"] == "number":
            numeric(answer["value"])
        elif not isinstance(answer["value"], str):
            raise ValueError("invalid enum answer")
    if not answers_agree(record["reference_answer"], record["oracle_answer"], spec):
        raise ValueError("reference answer disagrees with independent formula")


def load_json(path) -> dict:
    def unique_object(pairs):
        obj = {}
        for key, item in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = item
        return obj

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

"""Development-only paired immutable worlds; no mid-rollout mutation or test use."""

import copy
from collections import Counter

from .research_answers import answer_specification, answers_equal, oracle_answer
from .research_checker import call_signature, check_attempt, protocol_plan
from .research_environment import ResearchEnvironment
from .research_schema import digest, model_messages

INTERVENTION_VERSION = "research-observation-pairs-v1"
DYNAMIC_RECIPES = frozenset({
    "age_calculation", "weather_convert", "population_math", "search_and_convert",
    "country_density_convert", "gdp_scale", "weather_offset_convert", "measurement_scale", "distance_offset_convert",
})


def diagnostic_view(record: dict) -> dict:
    """Not a dataset-schema record: deliberately omit stale golds/references/signatures."""
    keys = {"sample_id", "semantic_group_id", "split", "recipe_id", "family", "task", "parameters",
            "environment_snapshot", "environment_hash", "generalization", "reference_depth"}
    return {key: copy.deepcopy(record[key]) for key in keys}


def make_pair(record: dict) -> dict:
    if record["split"] not in {"train", "validation"}:
        raise ValueError("observation interventions are development-only; test is forbidden")
    recipe = record["recipe_id"]
    if recipe not in DYNAMIC_RECIPES:
        raise ValueError("recipe has no reference dynamic dependency")
    base = diagnostic_view(record)
    variant = copy.deepcopy(base)
    entity = base["parameters"]["entity"]
    if recipe in {"age_calculation", "search_and_convert", "measurement_scale", "distance_offset_convert"}:
        table, field = "search", "birth_year" if recipe == "age_calculation" else "distance_km"
        fact = variant["environment_snapshot"][table][entity][field]
        before = fact["value"]
        fact["value"] = after = before + 7
    elif recipe in {"weather_convert", "weather_offset_convert"}:
        table, field = "weather", "temperature"
        fact = variant["environment_snapshot"][table][entity]
        before = fact[field]
        fact[field] = after = round(before + 7, 2)
    else:
        table, field = "countries", "gdp_billion_usd" if recipe == "gdp_scale" else "population"
        fact = variant["environment_snapshot"][table][entity]
        before = fact[field]
        fact[field] = after = before + (7 if field == "gdp_billion_usd" else max(1, round(fact["area_km2"] * 7)))
    ResearchEnvironment(variant["environment_snapshot"])
    variant["environment_hash"] = digest(variant["environment_snapshot"])
    if model_messages(base) != model_messages(variant):
        raise ValueError("intervention changed public prompt")
    a, edges = protocol_plan(base)
    b, variant_edges = protocol_plan(variant)
    consumers = sorted({target for _, target in edges})
    changed = [i for i, (x, y) in enumerate(zip(a, b, strict=True))
               if call_signature(x["name"], x["arguments"]) != call_signature(y["name"], y["arguments"])]
    if edges != variant_edges or changed != consumers or answers_equal(oracle_answer(base), oracle_answer(variant), answer_specification(base)):
        raise ValueError("ineffective or invalid paired intervention")
    return {"version": INTERVENTION_VERSION, "group_id": record["semantic_group_id"],
            "source_sample_id": record["sample_id"], "source_split": record["split"],
            "base": base, "variant": variant, "intervention": {"table": table, "entity": entity, "field": field,
                                                                   "before": before, "after": after},
            "consumer_indices": consumers}


def build_pairs(records: list[dict]) -> list[dict]:
    if any(row["split"] not in {"train", "validation"} for row in records):
        raise ValueError("test input forbidden, even if no dynamic rows would be selected")
    pairs = [make_pair(row) for row in records if row["recipe_id"] in DYNAMIC_RECIPES]
    if len({p["group_id"] for p in pairs}) != len(pairs):
        raise ValueError("duplicate diagnostic group")
    return pairs


def check_pair(pair: dict, base_attempt: dict, variant_attempt: dict) -> dict:
    """Both correct + truthful routes + changed consumed parameters/answers.

    Alternative Fahrenheit queries bypass the edge: still require changed real
    producer observations and answers, but report bypass separately. This is a
    diagnostic on a trusted flat trace, not proof of causal attention by a model.
    """
    results = [check_attempt(pair[side], attempt["content"], attempt["trace"],
                             termination=attempt.get("termination", "finished"))
               for side, attempt in (("base", base_attempt), ("variant", variant_attempt))]
    a, b = results
    both_answers = a["answer"]["correct"] and b["answer"]["correct"]
    both_routes = a["trace"]["semantic_valid"] and b["trace"]["semantic_valid"]
    answer_changed = False
    if a["answer"]["parsed"] is not None and b["answer"]["parsed"] is not None:
        answer_changed = not answers_equal(a["answer"]["parsed"], b["answer"]["parsed"], answer_specification(pair["base"]))
    dependency_valid = all(r["trace"]["dynamic_status"] == "valid" for r in results)
    bypass = all(r["trace"]["dynamic_status"] == "alternative_bypass" for r in results)
    parameters_changed = False
    if dependency_valid:
        parameters_changed = all(
            call_signature(base_attempt["trace"][a["trace"]["matched_steps"][i]]["name"],
                           base_attempt["trace"][a["trace"]["matched_steps"][i]]["arguments"])
            != call_signature(variant_attempt["trace"][b["trace"]["matched_steps"][i]]["name"],
                              variant_attempt["trace"][b["trace"]["matched_steps"][i]]["arguments"])
            for i in pair["consumer_indices"]
        )
    responsive = both_answers and both_routes and answer_changed and (parameters_changed or bypass)
    return {"version": INTERVENTION_VERSION, "group_id": pair["group_id"], "base": a, "variant": b,
            "both_answers_correct": both_answers, "answer_changed": answer_changed,
            "both_routes_valid": both_routes, "consumers_changed": parameters_changed,
            "dependency_valid": dependency_valid, "alternative_bypass": bypass, "responsive": responsive}


def summarize_pairs(pairs: list[dict], outcomes: list[dict]) -> dict:
    """Group denominator never shrinks; missing attempts remain explicit failures."""
    expected = {p["group_id"] for p in pairs}
    ids = [r["group_id"] for r in outcomes]
    if len(set(ids)) != len(ids) or not set(ids) <= expected:
        raise ValueError("duplicate/unknown paired result")
    successes = sum(r["responsive"] for r in outcomes)
    return {"planned_groups": len(pairs), "returned_groups": len(outcomes), "missing_groups": len(pairs) - len(outcomes),
            "responsive_groups": successes, "responsive_rate": successes / len(pairs) if pairs else None,
            "consumer_changed_groups": sum(r["consumers_changed"] for r in outcomes),
            "alternative_bypass_groups": sum(r["alternative_bypass"] for r in outcomes),
            "source_split_counts": dict(Counter(p["source_split"] for p in pairs)),
            "recipe_counts": dict(Counter(p["base"]["recipe_id"] for p in pairs))}

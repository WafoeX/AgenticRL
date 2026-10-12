"""Fail-closed CPU capacity, identity, execution and split-policy audit."""

import json
from collections import Counter
from itertools import combinations

from .research_generator import (
    entity_name,
    family_quotas,
    make_instance,
    render_record,
    validate_config,
)
from .research_schema import digest, model_messages, normalized_text, validate_record


def observation_facts(row: dict, *, required_only: bool) -> set:
    """Facts that were visible in real observations, not the whole hidden snapshot."""
    facts = set()
    for step in row["reference_trace"]:
        name, arguments = step["name"], step["arguments"]
        if name not in {"search", "get_weather", "lookup_db"}:
            continue
        try:
            observation = json.loads(step["observation"])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid reference observation") from exc
        if name == "search":
            fact = observation[0]
            facts.add(digest([name, fact["entity"], fact["field"], fact["value"], fact["unit"]]))
            continue
        entity = arguments["city"] if name == "get_weather" else arguments["key"]
        if required_only:
            fields = (["temperature"] if name == "get_weather" else
                      ["population"] if row["recipe_id"] == "country_compare" else
                      ["population", "area_km2"] if row["recipe_id"] in {"population_math", "country_density_convert"} else
                      ["gdp_billion_usd"])
        else:
            fields = list(observation)
        for field in fields:
            facts.add(digest([name, entity, field, observation[field]]))
    return facts


def audit_dataset(data: dict[str, list], config: dict) -> dict:
    validate_config(config)
    if set(data) != {"train", "validation", "test"}:
        raise ValueError("missing or extra dataset split")
    expected_counts = Counter()
    buckets = {}
    for bucket in config["buckets"]:
        buckets[bucket["split"], bucket["generalization"]] = bucket
        expected_counts.update({(bucket["split"], bucket["generalization"], recipe): count
                                for recipe, count in family_quotas(config, bucket).items() if count})
    actual_counts = Counter()
    comparisons = {}
    summaries = {}
    all_ids = set()
    for split, rows in data.items():
        if not isinstance(rows, list):
            raise ValueError("split must be a list")
        for row in rows:
            validate_record(row)
            if row["split"] != split or row["sample_id"] in all_ids:
                raise ValueError("split mismatch or duplicate sample ID")
            all_ids.add(row["sample_id"])
            key = (split, row["generalization"])
            if key not in buckets:
                raise ValueError("unregistered generalization bucket")
            bucket = buckets[key]
            index = row["candidate_index"]
            if not 0 <= index < config["candidate_pool_per_recipe_domain"]:
                raise ValueError("candidate outside registered domain")
            recipe = row["recipe_id"]
            if recipe not in family_quotas(config, bucket):
                raise ValueError("reserved recipe leaked into wrong split/bucket")
            winner_a = recipe in {"weather_compare", "country_compare"} and row["parameters"]["entity"] == entity_name(config, index + 1024)
            # Re-derive every hidden fact, signature, dependency and gold; do not trust stored hashes.
            expected = render_record(config, make_instance(config, recipe, bucket["domain"], index, winner_a=winner_a), bucket)
            if row != expected:
                raise ValueError("record differs from registered instance/wording/reference construction")
            actual_counts[split, row["generalization"], recipe] += 1
            if recipe in {"weather_compare", "country_compare"}:
                balance = comparisons.setdefault(f"{split}:{row['generalization']}:{recipe}", Counter())
                answer = row["oracle_answer"]["value"]
                slot = "tie" if answer == "tie" else "A" if answer == row["parameters"]["entity"] else "B"
                balance[slot] += 1
            messages = model_messages(row)
            if messages[1] != {"role": "user", "content": row["task"]} or len(messages) != 2:
                raise ValueError("model message boundary violated")
        summaries[split] = {
            "rows": len(rows),
            "depth_counts": dict(sorted(Counter(str(r["reference_depth"]) for r in rows).items())),
            "family_counts": dict(sorted(Counter(r["family"] for r in rows).items())),
            "generalization_counts": dict(sorted(Counter(r["generalization"] for r in rows).items())),
            "dynamic_dependency_rows": sum(r["dynamic_dependency"] for r in rows),
        }
    if expected_counts != actual_counts:
        raise ValueError(f"quota mismatch: expected {dict(expected_counts)}, got {dict(actual_counts)}")
    for balance in comparisons.values():
        if balance["tie"] or abs(balance["A"] - balance["B"]) > 1:
            raise ValueError("comparison slot imbalance (formal main set excludes ties)")
    signatures = {
        "exact_text": lambda r: r["task"], "normalized_text": lambda r: normalized_text(r["task"]),
        "semantic_group": lambda r: r["semantic_group_id"], "value_signature": lambda r: r["value_signature"],
        "trajectory_signature": lambda r: r["trajectory_signature"],
    }
    sets = {}
    for split, rows in data.items():
        sets[split] = {name: {extract(r) for r in rows} for name, extract in signatures.items()}
        for name, values in sets[split].items():
            if len(values) != len(rows):
                raise ValueError(f"within-split duplicate {split}:{name}")
    overlaps = {}
    for left, right in combinations(data, 2):
        overlaps[f"{left}:{right}"] = {name: len(sets[left][name] & sets[right][name]) for name in signatures}
        if any(overlaps[f"{left}:{right}"].values()):
            raise ValueError(f"cross-split leakage: {left}:{right}")
    # Words/parameters may share base domains for IID, wording and composition; only designated
    # OOD buckets require numeric domain disjointness. Wording holdout IDs are explicitly separate.
    train_entities = {v for r in data["train"] for k, v in r["parameters"].items() if k in {"entity", "entity_b"}}
    novelty = {}
    for split in ("validation", "test"):
        novelty[split] = {}
        for kind in ("iid", "parameter", "wording", "composition"):
            rows = [r for r in data[split] if r["generalization"] == kind]
            entities = {v for r in rows for k, v in r["parameters"].items() if k in {"entity", "entity_b"}}
            novelty[split][kind] = {"entity_count": len(entities), "unseen_in_train": len(entities - train_entities)}
    train_observed_facts = set().union(*(observation_facts(r, required_only=False) for r in data["train"]))
    fact_exposure = {}
    for split in ("validation", "test"):
        fact_exposure[split] = {}
        for kind in ("iid", "parameter", "wording", "composition"):
            required_sets = [observation_facts(r, required_only=True) for r in data[split] if r["generalization"] == kind]
            hidden = [facts for facts in required_sets if facts]
            fact_exposure[split][kind] = {
                "rows_requiring_hidden_facts": len(hidden),
                "rows_with_any_fact_previously_observed_in_train": sum(bool(f & train_observed_facts) for f in hidden),
                "rows_with_all_facts_previously_observed_in_train": sum(f <= train_observed_facts for f in hidden),
            }
    train_sequences = {tuple(c["name"] for c in r["expected_calls"]) for r in data["train"]}
    composition = {}
    train_tools = {c["name"] for r in data["train"] for c in r["expected_calls"]}
    train_operations = {r["parameters"]["op"] for r in data["train"] if r["recipe_id"] == "simple_math"}
    train_conversions = {(r["parameters"]["from_unit"], r["parameters"]["to_unit"])
                         for r in data["train"] if r["recipe_id"] == "simple_convert"}
    for split in ("validation", "test"):
        recipes = sorted({r["recipe_id"] for r in data[split] if r["generalization"] == "composition"})
        composition[split] = {}
        for recipe in recipes:
            tools = config["recipes"][recipe]["tools"]
            operation = "*" if recipe in {"gdp_scale", "measurement_scale"} else "+"
            conversion = (("celsius", "fahrenheit") if recipe == "weather_offset_convert" else
                          ("km", "miles") if recipe == "distance_offset_convert" else None)
            if not set(tools) <= train_tools or operation not in train_operations or (conversion and conversion not in train_conversions):
                raise ValueError("composition contains unseen atomic tool/operation/conversion")
            composition[split][recipe] = {"tool_sequence": tools, "sequence_seen_in_train": tuple(tools) in train_sequences,
                                          "atomic_tools_seen_in_train": True, "arithmetic_operation_seen_in_train": operation,
                                          "conversion_seen_in_train": list(conversion) if conversion else None,
                                          "semantic_recipe_seen_in_train": False}
    return {"audit_version": "research-split-audit-v1", "status": "pass", "config_hash": digest(config),
            "datasets": summaries, "cross_split_overlap": overlaps,
            "comparison_balance": {k: dict(v) for k, v in sorted(comparisons.items())},
            "composition_holdout": composition, "entity_novelty": novelty, "train_observation_fact_exposure": fact_exposure,
            "replayed_references": sum(len(rows) for rows in data.values()),
            "frozen_test": False,
            "limitations": ["Candidate data; R1.6 intervention fixtures and R2 model-answer checker not implemented.",
                            "All snapshots are synthetic and scenario-local; shared facts are not independent external truth.",
                            "Entity novelty is disclosed; these are not pure numerical-only OOD experiments.",
                            "Some IID/wording/composition facts were observable in train; observation-fact exposure is diagnostic, not equivalent-task label overlap.",
                            "Exact/normalized text and declared semantic/trajectory identities are audited, not all linguistic paraphrases.",
                            "weather_convert admits a legal one-call Fahrenheit solution; depth is reference depth.",
                            "Main-set comparisons exclude ties; tie handling is covered by CPU fixtures."]}

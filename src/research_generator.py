"""Finite, group-first Research V1 allocation and executed reference construction."""

import ast
import copy
import json
import random
from collections import Counter
from decimal import ROUND_HALF_EVEN, Decimal

from .research_environment import (
    CONVERSION_PAIRS,
    ENVIRONMENT_VERSION,
    ResearchEnvironment,
    canonical_json,
)
from .research_schema import SCHEMA_VERSION, digest, normalized_text, validate_record

RECIPE_TOOLS = {
    "simple_math": ["calculator"], "simple_weather": ["get_weather"], "simple_search": ["search"],
    "simple_convert": ["convert_units"], "simple_lookup": ["lookup_db"],
    "age_calculation": ["search", "calculator"], "weather_convert": ["get_weather", "convert_units"],
    "population_math": ["lookup_db", "calculator"], "search_and_convert": ["search", "convert_units"],
    "weather_compare": ["get_weather", "get_weather"], "country_compare": ["lookup_db", "lookup_db"],
    "country_density_convert": ["lookup_db", "calculator", "convert_units"],
    "gdp_scale": ["lookup_db", "calculator"], "weather_offset_convert": ["get_weather", "calculator", "convert_units"],
    "measurement_scale": ["search", "calculator"], "distance_offset_convert": ["search", "calculator", "convert_units"],
}
PARTITIONS = {r: ("validation" if r in {"gdp_scale", "weather_offset_convert"} else
                  "test" if r in {"measurement_scale", "distance_offset_convert"} else "train") for r in RECIPE_TOOLS}


def validate_config(config: dict) -> None:
    if set(config) != {"dataset_version", "seed", "candidate_pool_per_recipe_domain", "domains", "buckets", "recipes"}:
        raise ValueError("invalid config fields")
    if type(config["seed"]) is not int or config["seed"] < 0 or type(config["candidate_pool_per_recipe_domain"]) is not int or not 1 <= config["candidate_pool_per_recipe_domain"] <= 1024:
        raise ValueError("invalid seed/candidate pool (maximum 1024)")
    if not isinstance(config["dataset_version"], str) or not config["dataset_version"]:
        raise ValueError("invalid dataset version")
    if set(config["recipes"]) != set(RECIPE_TOOLS) or set(config["domains"]) != {"base", "val_ood", "test_ood"}:
        raise ValueError("unknown recipe/domain registry")
    keys = {"operand_base", "temperature_base", "population_base", "area_base", "gdp_base", "birth_base",
            "target_base", "distance_base", "conversion_base", "scale_base", "offset_base"}
    for domain in config["domains"].values():
        if set(domain) != keys or not all(type(v) is int for v in domain.values()):
            raise ValueError("invalid domain fields")
        if min(domain[k] for k in keys - {"temperature_base"}) <= 0:
            raise ValueError("nonpositive parameter domain")
    # Every numeric family has a concrete, finite domain. Enforce mutually disjoint ranges.
    widths = {"operand_base": 1023, "temperature_base": 25.23, "population_base": 264690,
              "area_base": 96, "gdp_base": 3069, "birth_base": 79, "target_base": 12,
              "distance_base": 102.3, "conversion_base": 1023, "scale_base": 4, "offset_base": 6}
    for key, width in widths.items():
        intervals = sorted((d[key], d[key] + width) for d in config["domains"].values())
        if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:], strict=False)):
            raise ValueError(f"overlapping parameter domain: {key}")
    for recipe, entry in config["recipes"].items():
        if set(entry) != {"depth", "partition", "tools", "wordings"} or entry["tools"] != RECIPE_TOOLS[recipe] or entry["depth"] != len(entry["tools"]) or entry["partition"] != PARTITIONS[recipe]:
            raise ValueError("recipe registration differs from implementation")
        if len(entry["wordings"]) != 3 or len(set(entry["wordings"])) != 3:
            raise ValueError("wording registry must contain three distinct patterns")
    seen = set()
    for bucket in config["buckets"]:
        if set(bucket) != {"split", "generalization", "domain", "wording", "depth_counts"}:
            raise ValueError("invalid bucket fields")
        split, kind = bucket["split"], bucket["generalization"]
        if split not in {"train", "validation", "test"} or kind not in {"train", "iid", "parameter", "wording", "composition"} or (split == "train") != (kind == "train"):
            raise ValueError("invalid bucket split/kind")
        if (split, kind) in seen:
            raise ValueError("duplicate bucket")
        seen.add((split, kind))
        domain = "val_ood" if split == "validation" else "test_ood"
        if bucket["domain"] != (domain if kind == "parameter" else "base") or bucket["wording"] != (split if kind == "wording" else "train"):
            raise ValueError("holdout domain/wording contract violation")
        if len(bucket["depth_counts"]) != 3 or not all(type(n) is int and n >= 0 for n in bucket["depth_counts"]):
            raise ValueError("invalid depth quota")
        if kind == "composition" and bucket["depth_counts"][0]:
            raise ValueError("composition cannot have depth 1")


def family_quotas(config: dict, bucket: dict) -> dict[str, int]:
    partition = bucket["split"] if bucket["generalization"] == "composition" else "train"
    quotas = {}
    for depth, total in enumerate(bucket["depth_counts"], 1):
        recipes = sorted(r for r, e in config["recipes"].items() if e["depth"] == depth and e["partition"] == partition)
        if total and not recipes:
            raise ValueError("no recipes for quota")
        for i, recipe in enumerate(recipes):
            quotas[recipe] = total // len(recipes) + (i < total % len(recipes))
    return quotas


def entity_name(config: dict, index: int) -> str:
    """Shared anonymous entity pool, with no visible arithmetic link to fact index."""
    return "Fiction-" + digest(["entity-alias-v1", config["seed"], index])[:16]


def make_instance(config: dict, recipe: str, domain: str, index: int, *, winner_a: bool = False) -> dict:
    """Finite candidate: useful facts are indexed injectively; only distractors use local RNG."""
    d = config["domains"][domain]
    seed = int(digest([config["seed"], recipe, domain, index])[:16], 16)
    rng = random.Random(seed)
    entity = entity_name(config, index)
    entity_b = entity_name(config, index + 1024)
    weather = {"temperature": round(d["temperature_base"] + index / 100, 2),
               "condition": rng.choice(["sunny", "cloudy", "rainy"]), "humidity": rng.randint(20, 90)}
    country = {"population": d["population_base"] + index * 137,
               "area_km2": d["area_base"] + index % 97, "gdp_billion_usd": d["gdp_base"] + index * 3}
    facts = {"birth_year": {"value": d["birth_base"] + index % 80, "unit": "years"},
             "distance_km": {"value": round(d["distance_base"] + index / 10, 1), "unit": "km"}}
    world = {"version": ENVIRONMENT_VERSION, "weather": {entity: weather}, "countries": {entity: country}, "search": {entity: facts}}
    p = {"entity": entity}
    useful = {}
    if recipe == "simple_math":
        p = {"a": d["operand_base"] + index, "b": d["operand_base"] + index % 97, "op": ["+", "-", "*"][index % 3]}
    elif recipe == "simple_convert":
        source, target = CONVERSION_PAIRS[index % len(CONVERSION_PAIRS)]
        p = {"value": d["conversion_base"] + index, "from_unit": source, "to_unit": target}
    elif recipe == "age_calculation":
        p["year"] = d["target_base"] + index // 80
        useful = {"birth_year": facts["birth_year"]["value"]}
    elif recipe in {"simple_weather", "weather_convert", "weather_offset_convert"}:
        useful = {"temperature": weather["temperature"]}
    elif recipe in {"simple_search", "search_and_convert", "measurement_scale", "distance_offset_convert"}:
        useful = {"distance_km": facts["distance_km"]["value"]}
    elif recipe in {"simple_lookup", "gdp_scale"}:
        useful = {"gdp_billion_usd": country["gdp_billion_usd"]}
    elif recipe in {"population_math", "country_density_convert"}:
        useful = {k: country[k] for k in ("population", "area_km2")}
    elif recipe in {"weather_compare", "country_compare"}:
        second_weather = {**weather, "temperature": round(weather["temperature"] + 15, 2)}
        second_country = {**country, "population": country["population"] + 100000 + index * 23}
        world["weather"][entity_b] = second_weather
        world["countries"][entity_b] = second_country
        p["entity_b"] = entity_b
        field = "temperature" if recipe == "weather_compare" else "population"
        useful = {"left": weather[field] if field == "temperature" else country[field],
                  "right": second_weather[field] if field == "temperature" else second_country[field]}
        if winner_a:
            p["entity"], p["entity_b"] = p["entity_b"], p["entity"]
            useful["left"], useful["right"] = useful["right"], useful["left"]
    else:
        raise ValueError("unimplemented recipe")
    if recipe in {"gdp_scale", "measurement_scale"}:
        p["scale"] = d["scale_base"] + index % 5
    if recipe in {"weather_offset_convert", "distance_offset_convert"}:
        p["offset"] = d["offset_base"] + index % 7
    deidentified = {k: v for k, v in p.items() if k not in {"entity", "entity_b"}}
    if recipe == "simple_math" and p["op"] in {"+", "*"}:
        deidentified["a"], deidentified["b"] = sorted((p["a"], p["b"]))
    semantic_p = dict(p)
    if recipe == "simple_math":
        semantic_p = deidentified.copy()
    semantic_facts = useful.copy()
    value_facts = useful.copy()
    if recipe in {"weather_compare", "country_compare"}:
        semantic_p = {"entities": sorted([[p["entity"], useful["left"]], [p["entity_b"], useful["right"]]])}
        semantic_facts = {}
        value_facts = {"compared_values": sorted(useful.values())}
    return {"recipe": recipe, "domain": domain, "index": index, "seed": seed, "parameters": p,
            "world": world, "useful": useful,
            "semantic_payload": {"operation": recipe, "inputs": semantic_p, "facts": semantic_facts},
            "value_payload": {"operation": recipe, "inputs": deidentified, "facts": value_facts}}


def formula_conversion(value, source: str, target: str) -> float:
    """Independent Decimal formulas, not imported tool conversion code."""
    v = Decimal(str(value))
    ratios = {("km", "miles"): (Decimal(1), Decimal("1.609344")),
              ("miles", "km"): (Decimal("1.609344"), Decimal(1)),
              ("kg", "lbs"): (Decimal(1), Decimal("0.45359237")),
              ("lbs", "kg"): (Decimal("0.45359237"), Decimal(1)),
              ("meters", "feet"): (Decimal(1), Decimal("0.3048")),
              ("feet", "meters"): (Decimal("0.3048"), Decimal(1)),
              ("people_per_km2", "people_per_mile2"): (Decimal("1.609344") ** 2, Decimal(1)),
              ("people_per_mile2", "people_per_km2"): (Decimal(1), Decimal("1.609344") ** 2)}
    if (source, target) == ("celsius", "fahrenheit"):
        result = v * Decimal("1.8") + 32
    elif (source, target) == ("fahrenheit", "celsius"):
        result = (v - 32) / Decimal("1.8")
    else:
        numerator, denominator = ratios[source, target]
        result = v * numerator / denominator
    return float(result.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))


def oracle_formula(instance: dict) -> dict:
    p, facts, recipe = instance["parameters"], instance["useful"], instance["recipe"]
    if recipe == "simple_math":
        a, b = Decimal(p["a"]), Decimal(p["b"])
        value = {"+": a + b, "-": a - b, "*": a * b}[p["op"]]
        return {"value": float(value), "unit": "dimensionless"}
    if recipe == "simple_convert":
        return {"value": formula_conversion(p["value"], p["from_unit"], p["to_unit"]), "unit": p["to_unit"]}
    if recipe == "age_calculation":
        return {"value": p["year"] - facts["birth_year"], "unit": "years"}
    if recipe in {"weather_compare", "country_compare"}:
        return {"value": "tie" if facts["left"] == facts["right"] else p["entity"] if facts["left"] > facts["right"] else p["entity_b"], "unit": "entity"}
    if recipe in {"simple_weather", "weather_convert", "weather_offset_convert"}:
        value = Decimal(str(facts["temperature"])) + p.get("offset", 0)
        return {"value": float(value) if recipe == "simple_weather" else formula_conversion(value, "celsius", "fahrenheit"), "unit": "celsius" if recipe == "simple_weather" else "fahrenheit"}
    if recipe in {"simple_search", "search_and_convert", "measurement_scale", "distance_offset_convert"}:
        value = Decimal(str(facts["distance_km"])) * p.get("scale", 1) + p.get("offset", 0)
        return {"value": float(value) if recipe in {"simple_search", "measurement_scale"} else formula_conversion(value, "km", "miles"), "unit": "km" if recipe in {"simple_search", "measurement_scale"} else "miles"}
    if recipe in {"simple_lookup", "gdp_scale"}:
        return {"value": facts["gdp_billion_usd"] * p.get("scale", 1), "unit": "billion_usd"}
    density = Decimal(facts["population"]) / Decimal(facts["area_km2"])
    return {"value": float(density) if recipe == "population_math" else formula_conversion(density, "people_per_km2", "people_per_mile2"), "unit": "people_per_km2" if recipe == "population_math" else "people_per_mile2"}


def reference_unit(recipe: str, parameters: dict) -> str:
    """Public recipe output contract; no oracle computation or hidden facts."""
    return {
        "simple_math": "dimensionless", "simple_weather": "celsius", "simple_search": "km",
        "simple_convert": parameters.get("to_unit", "dimensionless"), "simple_lookup": "billion_usd",
        "age_calculation": "years", "weather_convert": "fahrenheit", "population_math": "people_per_km2",
        "search_and_convert": "miles", "weather_compare": "entity", "country_compare": "entity",
        "country_density_convert": "people_per_mile2", "gdp_scale": "billion_usd",
        "weather_offset_convert": "fahrenheit", "measurement_scale": "km", "distance_offset_convert": "miles",
    }[recipe]


def execute_reference(instance: dict) -> tuple[list, list, dict]:
    """Construct each dependent argument ONLY after reading its producer observation."""
    recipe, p = instance["recipe"], instance["parameters"]
    env = ResearchEnvironment(instance["world"])
    trace, edges = [], []

    def call(name, arguments):
        observation = env.execute(name, arguments)
        if observation.startswith("Error:"):
            raise ValueError(f"unexecutable reference: {observation}")
        trace.append({"turn": len(trace), "name": name, "arguments": arguments, "observation": observation})
        # execute emits validated JSON, but fail closed if that contract ever changes.
        try:
            return json.loads(observation)
        except json.JSONDecodeError as exc:
            raise ValueError("tool produced non-JSON observation") from exc

    def edge(source, target, fields, argument, transform="identity"):
        edges.append({"source": source, "target": target, "source_fields": fields,
                      "target_argument": argument, "transform": transform})

    def conversion(value, source, target):
        return call("convert_units", {"value": value, "from_unit": source, "to_unit": target})

    unit = reference_unit(recipe, p)
    if recipe == "simple_math":
        result = call("calculator", {"expression": f"{p['a']}{p['op']}{p['b']}"})
    elif recipe == "simple_convert":
        result = conversion(p["value"], p["from_unit"], p["to_unit"])
    elif recipe in {"simple_weather", "weather_convert", "weather_offset_convert"}:
        result = call("get_weather", {"city": p["entity"], "units": "celsius"})["temperature"]
        if recipe == "weather_offset_convert":
            result = call("calculator", {"expression": f"{result}+{p['offset']}"})
            edge(0, 1, ["temperature"], "expression", "add_public_offset")
        if recipe != "simple_weather":
            source = len(trace) - 1
            result = conversion(result, "celsius", "fahrenheit")
            edge(source, source + 1, ["temperature" if source == 0 else "$"], "value")
    elif recipe in {"simple_search", "age_calculation", "search_and_convert", "measurement_scale", "distance_offset_convert"}:
        field = "birth_year" if recipe == "age_calculation" else "distance_km"
        result = call("search", {"query": f"{p['entity']} {field}"})[0]["value"]
        if recipe in {"age_calculation", "measurement_scale", "distance_offset_convert"}:
            expression = (f"{p['year']}-{result}" if recipe == "age_calculation" else
                          f"{result}*{p['scale']}" if recipe == "measurement_scale" else f"{result}+{p['offset']}")
            result = call("calculator", {"expression": expression})
            edge(0, 1, ["0.value"], "expression", {"age_calculation": "public_year_minus", "measurement_scale": "multiply_public_scale", "distance_offset_convert": "add_public_offset"}[recipe])
        if recipe in {"search_and_convert", "distance_offset_convert"}:
            source = len(trace) - 1
            result = conversion(result, "km", "miles")
            edge(source, source + 1, ["0.value" if source == 0 else "$"], "value")
    elif recipe in {"simple_lookup", "population_math", "country_density_convert", "gdp_scale"}:
        record = call("lookup_db", {"table": "countries", "key": p["entity"]})
        result = record["gdp_billion_usd"]
        if recipe in {"population_math", "country_density_convert"}:
            result = call("calculator", {"expression": f"{record['population']}/{record['area_km2']}"})
            edge(0, 1, ["population", "area_km2"], "expression", "divide_population_area")
        if recipe == "gdp_scale":
            result = call("calculator", {"expression": f"{result}*{p['scale']}"})
            edge(0, 1, ["gdp_billion_usd"], "expression", "multiply_public_scale")
        if recipe == "country_density_convert":
            result = conversion(result, "people_per_km2", "people_per_mile2")
            edge(1, 2, ["$"], "value")
    else:
        name = "get_weather" if recipe == "weather_compare" else "lookup_db"
        field = "temperature" if recipe == "weather_compare" else "population"
        results = []
        for entity in (p["entity"], p["entity_b"]):
            args = {"city": entity, "units": "celsius"} if name == "get_weather" else {"table": "countries", "key": entity}
            results.append(call(name, args)[field])
        result = "tie" if results[0] == results[1] else p["entity"] if results[0] > results[1] else p["entity_b"]
    return trace, edges, {"value": result, "unit": unit}


def canonical_expression(expression: str) -> object:
    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return [type(node.op).__name__, visit(node.operand)]
        if isinstance(node, ast.BinOp):
            operands = [visit(node.left), visit(node.right)]
            if isinstance(node.op, (ast.Add, ast.Mult)):
                operands.sort(key=canonical_json)
            return [type(node.op).__name__, *operands]
        raise ValueError("non-arithmetic reference expression")
    return visit(ast.parse(expression, mode="eval").body)


def render_record(config: dict, instance: dict, bucket: dict) -> dict:
    recipe = instance["recipe"]
    mode = {"train": 0, "validation": 1, "test": 2}[bucket["wording"]]
    trace, edges, reference = execute_reference(instance)
    calls = [{"name": t["name"], "arguments": t["arguments"]} for t in trace]
    deidentified_calls = copy.deepcopy(calls)
    for call in deidentified_calls:
        args = call["arguments"]
        for key in ("city", "key", "query"):
            if key in args:
                # Entity identity is not an escape hatch for duplicate numeric problems.
                args[key] = args[key].replace(instance["parameters"].get("entity_b", "\0"), "ENTITY").replace(instance["parameters"].get("entity", "\0"), "ENTITY")
        if "expression" in args:
            args["expression"] = canonical_expression(args["expression"])
    if not edges:
        deidentified_calls.sort(key=canonical_json)
    trajectory_payload = [{"meaning": instance["value_payload"]}, *deidentified_calls]
    semantic_signature, value_signature = digest(instance["semantic_payload"]), digest(instance["value_payload"])
    oracle = oracle_formula(instance)
    integer = recipe in {"simple_math", "age_calculation", "simple_lookup", "gdp_scale"}
    record = {
        "schema_version": SCHEMA_VERSION, "dataset_version": config["dataset_version"],
        "sample_id": "rv1-" + semantic_signature, "family": recipe, "recipe_id": recipe,
        "task": config["recipes"][recipe]["wordings"][mode].format(**instance["parameters"]),
        "wording_id": f"{recipe}:{bucket['wording']}:v1", "semantic_group_id": semantic_signature,
        "semantic_signature": semantic_signature, "value_signature": value_signature,
        "trajectory_signature": digest(trajectory_payload), "split": bucket["split"],
        "generalization": bucket["generalization"], "parameters": instance["parameters"],
        "parameter_domain": instance["domain"], "candidate_index": instance["index"],
        "environment_seed": instance["seed"], "environment_version": ENVIRONMENT_VERSION,
        "environment_snapshot": instance["world"], "environment_hash": digest(instance["world"]),
        "reference_depth": len(trace), "dependency_graph": edges, "expected_calls": calls,
        "reference_trace": trace, "dynamic_dependency": bool(edges),
        "answer_spec": {"type": "enum" if oracle["unit"] == "entity" else "number", "unit": oracle["unit"],
                        "abs_tol": 0 if integer or oracle["unit"] == "entity" else 0.000001,
                        "rel_tol": 0 if integer or oracle["unit"] == "entity" else 1e-9},
        "oracle_answer": oracle, "reference_answer": reference,
        "semantic_payload": instance["semantic_payload"], "value_payload": instance["value_payload"],
        "trajectory_payload": trajectory_payload,
    }
    validate_record(record)
    return record


def capacity_report(config: dict) -> dict:
    """Exhaust the declared finite indexed pools, counting deidentified semantic groups."""
    validate_config(config)
    demands = Counter()
    for bucket in config["buckets"]:
        demands.update({(r, bucket["domain"]): n for r, n in family_quotas(config, bucket).items()})
    report = {}
    for (recipe, domain), demand in sorted(demands.items()):
        if not demand:
            continue
        unique = {digest(make_instance(config, recipe, domain, i)["value_payload"])
                  for i in range(config["candidate_pool_per_recipe_domain"])}
        report[f"{recipe}:{domain}"] = {"enumerated_candidates": config["candidate_pool_per_recipe_domain"],
                                        "unique_value_groups": len(unique), "requested": demand,
                                        "remaining_lower_bound": len(unique) - demand}
        if len(unique) < demand:
            raise ValueError(f"insufficient capacity for {recipe}:{domain}: {len(unique)} < {demand}")
    return report


def generate_dataset(config: dict) -> tuple[dict[str, list], dict]:
    capacity = capacity_report(config)
    data = {"train": [], "validation": [], "test": []}
    seen_groups, seen_values, seen_text = set(), set(), set()
    rejected = Counter()
    quotas_log = []
    for bucket in config["buckets"]:
        quotas = family_quotas(config, bucket)
        quotas_log.append({**bucket, "family_counts": quotas})
        for recipe, count in sorted(quotas.items()):
            candidates = sorted(range(config["candidate_pool_per_recipe_domain"]),
                                key=lambda i: digest([config["seed"], recipe, bucket["domain"], i]))
            allocated = 0
            for index in candidates:
                if allocated == count:
                    break
                instance = make_instance(config, recipe, bucket["domain"], index, winner_a=allocated % 2 == 0)
                group, value = digest(instance["semantic_payload"]), digest(instance["value_payload"])
                if group in seen_groups or value in seen_values:
                    rejected["already_allocated_group"] += 1
                    continue
                mode = {"train": 0, "validation": 1, "test": 2}[bucket["wording"]]
                preview = config["recipes"][recipe]["wordings"][mode].format(**instance["parameters"])
                text = normalized_text(preview)
                if text in seen_text:
                    rejected["normalized_text_collision"] += 1
                    continue
                # Ownership is assigned to the whole semantic group before materializing the record.
                seen_groups.add(group)
                seen_values.add(value)
                seen_text.add(text)
                data[bucket["split"]].append(render_record(config, instance, bucket))
                allocated += 1
            if allocated != count:
                raise ValueError(f"capacity after grouping/text constraints: {bucket['split']} {recipe} {allocated}/{count}")
    return data, {"capacity": capacity, "quotas": quotas_log, "allocation_rejections": dict(rejected)}

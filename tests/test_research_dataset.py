"""CPU research-contract tests; no ART, model, network, or GPU imports."""

import copy
import hashlib
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import generate_research_dataset as cli
from src.research_audit import audit_dataset
from src.research_environment import (
    CONVERSION_PAIRS,
    TOOL_SCHEMAS,
    ResearchEnvironment,
    calculate,
    convert,
)
from src.research_generator import (
    RECIPE_TOOLS,
    canonical_expression,
    capacity_report,
    entity_name,
    execute_reference,
    generate_dataset,
    make_instance,
    oracle_formula,
    render_record,
    validate_config,
)
from src.research_schema import (
    RECORD_TYPES,
    SYSTEM_PROMPT,
    digest,
    load_json,
    model_messages,
    normalized_text,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_json(ROOT / "configs/research/dataset_v1.json")


@pytest.fixture(scope="module")
def generated(config):
    return generate_dataset(config)


@pytest.fixture(scope="module")
def audit(generated, config):
    return audit_dataset(generated[0], config)


def test_full_counts_depth_strata_and_capacity(generated, audit):
    data, allocation = generated
    assert {s: len(rows) for s, rows in data.items()} == {"train": 1200, "validation": 180, "test": 240}
    assert [audit["datasets"][s]["depth_counts"] for s in data] == [
        {"1": 360, "2": 480, "3": 360}, {"1": 54, "2": 72, "3": 54}, {"1": 72, "2": 96, "3": 72},
    ]
    assert audit["datasets"]["validation"]["generalization_counts"] == {"iid": 90, "parameter": 30, "wording": 30, "composition": 30}
    assert audit["datasets"]["test"]["generalization_counts"] == dict.fromkeys(["iid", "parameter", "wording", "composition"], 60)
    assert len(allocation["capacity"]) == 40
    assert min(c["remaining_lower_bound"] for c in allocation["capacity"].values()) == 613
    assert all(c["unique_value_groups"] == 1024 for c in allocation["capacity"].values())
    assert audit["replayed_references"] == 1620
    assert audit["frozen_test"] is False
    assert all(not any(pair.values()) for pair in audit["cross_split_overlap"].values())


@pytest.mark.parametrize("recipe", sorted(RECIPE_TOOLS))
def test_all_recipes_all_domains_executed_and_formula_checked(config, recipe):
    partition = config["recipes"][recipe]["partition"]
    for domain in config["domains"]:
        for index in (0, 1, 97, 1023):
            instance = make_instance(config, recipe, domain, index)
            # Recipe execution independently of formal split allocation.
            bucket = {"split": partition, "generalization": "train" if partition == "train" else "composition", "wording": "train"}
            row = render_record(config, instance, bucket)
            assert [c["name"] for c in row["expected_calls"]] == RECIPE_TOOLS[recipe]
            assert len(row["reference_trace"]) == config["recipes"][recipe]["depth"]
            assert "__dynamic__" not in json.dumps(row)
            assert bool(row["dependency_graph"]) == row["dynamic_dependency"]


def test_json_schema_envelope_matches_runtime_contract(generated):
    schema = load_json(ROOT / "configs/research/schema_v1.json")
    assert set(schema["required"]) == set(schema["properties"]) == set(RECORD_TYPES)
    assert schema["additionalProperties"] is False
    for rows in generated[0].values():
        for row in rows:
            validate_record(row, replay=False)


def test_snapshot_isolation_and_concurrency(config):
    first = make_instance(config, "weather_convert", "base", 1)
    second = make_instance(config, "weather_convert", "test_ood", 1)
    world = copy.deepcopy(first["world"])
    a, b = ResearchEnvironment(world), ResearchEnvironment(second["world"])
    world["weather"][first["parameters"]["entity"]]["temperature"] = 999
    exposed = a.snapshot
    exposed["weather"].clear()
    args = {"city": first["parameters"]["entity"]}
    before = (a.execute("get_weather", args), b.execute("get_weather", args))
    assert before[0] != before[1]
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda i: (a if i % 2 == 0 else b).execute("get_weather", args), range(100)))
    assert outputs == [before[i % 2] for i in range(100)]


@pytest.mark.parametrize("expression", ["__import__('os')", "2**10000000", "1/0", "True+2", "x+2", "1e999", "[1]", "1" * 257])
def test_calculator_fails_closed(config, expression):
    env = ResearchEnvironment(make_instance(config, "simple_math", "base", 0)["world"])
    assert env.execute("calculator", {"expression": expression}).startswith("Error:")


def test_arithmetic_and_conversion_hand_calculations():
    assert calculate("(3+5)*2-10/2") == 11
    assert convert(0, "celsius", "fahrenheit") == 32
    assert convert(212, "fahrenheit", "celsius") == 100
    assert convert(1.609344, "km", "miles") == 1
    assert convert(0.45359237, "kg", "lbs") == 1
    assert convert(0.3048, "meters", "feet") == 1
    assert convert(1, "people_per_km2", "people_per_mile2") == 2.589988
    assert convert(2.589988110336, "people_per_mile2", "people_per_km2") == 1


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "10", None])
def test_bad_numbers_rejected(config, value):
    env = ResearchEnvironment(make_instance(config, "simple_convert", "base", 0)["world"])
    assert env.execute("convert_units", {"value": value, "from_unit": "km", "to_unit": "miles"}).startswith("Error:")


def test_strict_tool_arguments_exact_search_and_missing_entity(config):
    env = ResearchEnvironment(make_instance(config, "simple_search", "base", 0)["world"])
    entity = entity_name(config, 0)
    for name, args in [("unknown", {}), ("calculator", {}), ("calculator", {"expression": "1", "extra": 1}),
                       ("lookup_db", {"table": "wrong", "key": entity}),
                       ("get_weather", {"city": entity, "units": "kelvin"}),
                       ("search", {"query": f"{entity} distance"}), ("search", {"query": "unmatched birth_year"})]:
        assert env.execute(name, args).startswith("Error:")
    fact = json.loads(env.execute("search", {"query": f"{entity} distance_km"}))
    assert fact == [{"entity": entity, "field": "distance_km", "value": 100.0, "unit": "km"}]


def test_balanced_comparisons_and_independent_queries(audit, generated):
    assert all(abs(c.get("A", 0) - c.get("B", 0)) <= 1 for c in audit["comparison_balance"].values())
    for rows in generated[0].values():
        for row in rows:
            if row["family"] in {"weather_compare", "country_compare"}:
                assert not row["dependency_graph"] and not row["dynamic_dependency"]


@pytest.mark.parametrize("recipe,field,table", [("weather_compare", "temperature", "weather"), ("country_compare", "population", "countries")])
def test_ties_have_no_slot_bias(config, recipe, field, table):
    instance = make_instance(config, recipe, "base", 4)
    p = instance["parameters"]
    instance["world"][table][p["entity_b"]][field] = instance["world"][table][p["entity"]][field]
    instance["useful"]["right"] = instance["useful"]["left"]
    assert execute_reference(instance)[2] == oracle_formula(instance) == {"value": "tie", "unit": "entity"}


def test_weather_fahrenheit_is_legal_one_call_alternative(config):
    instance = make_instance(config, "weather_convert", "base", 13)
    env = ResearchEnvironment(instance["world"])
    alternate = json.loads(env.execute("get_weather", {"city": instance["parameters"]["entity"], "units": "fahrenheit"}))
    assert alternate["temperature"] == oracle_formula(instance)["value"]


def test_reference_consumes_actual_observation_not_hidden_useful_values(config):
    instance = make_instance(config, "country_density_convert", "base", 0)
    entity = instance["parameters"]["entity"]
    instance["world"]["countries"][entity].update(population=1000, area_km2=10)
    trace, edges, answer = execute_reference(instance)
    assert trace[1]["arguments"] == {"expression": "1000/10"}
    assert trace[2]["arguments"]["value"] == 100
    assert answer == {"value": 258.998811, "unit": "people_per_mile2"}
    assert len(edges) == 2
    assert oracle_formula(instance) != answer  # stale useful/oracle facts do not steer execution


def test_normalization_commutativity_and_noncommutativity(config):
    assert normalized_text(" A   B?! ") == normalized_text("a b.")
    assert canonical_expression("3+4") == canonical_expression("4+3")
    assert canonical_expression("3*4") == canonical_expression("4*3")
    assert canonical_expression("3-4") != canonical_expression("4-3")
    assert canonical_expression("3/4") != canonical_expression("4/3")
    a = make_instance(config, "weather_compare", "base", 3)
    b = make_instance(config, "weather_compare", "base", 3, winner_a=True)
    assert a["semantic_payload"] == b["semantic_payload"] and a["value_payload"] == b["value_payload"]


def test_holdout_recipes_wordings_and_parameter_domains(audit, generated):
    data = generated[0]
    assert {r["recipe_id"] for r in data["train"]} == {r for r in RECIPE_TOOLS if r not in {"gdp_scale", "weather_offset_convert", "measurement_scale", "distance_offset_convert"}}
    assert audit["composition_holdout"]["test"]["measurement_scale"]["sequence_seen_in_train"] is True
    assert audit["composition_holdout"]["test"]["distance_offset_convert"]["sequence_seen_in_train"] is False
    assert audit["composition_holdout"]["validation"]["gdp_scale"]["sequence_seen_in_train"] is True
    for split in ("validation", "test"):
        for row in data[split]:
            if row["generalization"] == "wording":
                assert f":{split}:" in row["wording_id"]
            if row["generalization"] == "parameter":
                assert row["parameter_domain"] == ("val_ood" if split == "validation" else "test_ood")


def test_model_message_whitelist_does_not_leak_gold(generated):
    for rows in generated[0].values():
        for record in rows:
            messages = model_messages(record)
            assert messages == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": record["task"]}]
            for hidden in ["oracle_answer", "expected_calls", "reference_trace", "environment_snapshot", "semantic_payload"]:
                assert hidden not in json.dumps(messages)
            altered = copy.deepcopy(record)
            altered["oracle_answer"] = {"value": "SECRET", "unit": "entity"}
            altered["environment_snapshot"] = {"SECRET": 123}
            assert model_messages(altered) == messages


@pytest.mark.parametrize("mutation", ["extra", "missing", "bool_depth", "dynamic", "observation", "unit", "wrong_hash", "oracle", "graph_order"])
def test_record_contract_rejects_corruption(config, mutation):
    bucket = {"split": "train", "generalization": "train", "wording": "train"}
    row = render_record(config, make_instance(config, "country_density_convert", "base", 0), bucket)
    if mutation == "extra":
        row["extra"] = 1
    elif mutation == "missing":
        del row["sample_id"]
    elif mutation == "bool_depth":
        row["reference_depth"] = True
    elif mutation == "dynamic":
        row["expected_calls"][2]["arguments"]["value"] = "__dynamic__"
    elif mutation == "observation":
        row["reference_trace"][0]["observation"] = "{}"
    elif mutation == "unit":
        row["oracle_answer"]["unit"] = "miles"
    elif mutation == "wrong_hash":
        row["environment_hash"] = "0" * 64
    elif mutation == "oracle":
        row["oracle_answer"]["value"] = -1
    else:
        row["dependency_graph"][0]["source"] = 1
    with pytest.raises(ValueError):
        validate_record(row)


def test_audit_rederives_metadata_instead_of_trusting_rehashed_labels(config, generated):
    data = copy.deepcopy(generated[0])
    row = data["train"][0]
    row["oracle_answer"]["value"] += 1
    row["reference_answer"]["value"] += 1
    validate_record(row)  # agreement alone is NOT sufficient
    with pytest.raises(ValueError, match="registered instance"):
        audit_dataset(data, config)


def test_cross_split_same_semantic_instance_is_rejected(config, generated):
    data = copy.deepcopy(generated[0])
    original = next(r for r in data["train"] if r["recipe_id"] == "simple_weather")
    bucket = next(b for b in config["buckets"] if b["split"] == "test" and b["generalization"] == "iid")
    leaked = render_record(config, make_instance(config, "simple_weather", "base", original["candidate_index"]), bucket)
    target = next(i for i, r in enumerate(data["test"]) if r["recipe_id"] == "simple_weather" and r["generalization"] == "iid")
    data["test"][target] = leaked
    with pytest.raises(ValueError, match="duplicate sample ID"):
        audit_dataset(data, config)


def test_missing_samples_fail_instead_of_silent_denominator_shrinkage(config, generated):
    data = copy.deepcopy(generated[0])
    data["train"].pop()
    with pytest.raises(ValueError, match="quota mismatch"):
        audit_dataset(data, config)


@pytest.mark.parametrize("mutation", ["tiny_pool", "domain_overlap", "unknown_recipe", "bad_reserved", "bad_wording", "bad_depth", "bool_seed"])
def test_invalid_config_or_insufficient_capacity_fails(config, mutation):
    c = copy.deepcopy(config)
    if mutation == "tiny_pool":
        c["candidate_pool_per_recipe_domain"] = 10
    elif mutation == "domain_overlap":
        c["domains"]["val_ood"] = c["domains"]["base"].copy()
    elif mutation == "unknown_recipe":
        c["recipes"]["extra"] = c["recipes"]["simple_math"]
    elif mutation == "bad_reserved":
        c["recipes"]["measurement_scale"]["partition"] = "train"
    elif mutation == "bad_wording":
        c["buckets"][7]["wording"] = "train"
    elif mutation == "bool_seed":
        c["seed"] = True
    else:
        c["buckets"][4]["depth_counts"][0] = 1
    with pytest.raises(ValueError):
        capacity_report(c)


def test_generation_is_reproducible_and_preserves_global_rng(config, generated):
    random.seed(815)
    state = random.getstate()
    second = generate_dataset(config)
    assert random.getstate() == state
    assert digest(second) == digest(generated)


def test_candidate_identity_ignores_python_hash_seed(config):
    code = "from pathlib import Path; from src.research_schema import load_json,digest; from src.research_generator import make_instance; c=load_json(Path('configs/research/dataset_v1.json')); print(digest(make_instance(c,'country_density_convert','base',1023)))"
    import os
    results = []
    for seed in ("1", "99"):
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True)
        results.append(proc.stdout)
    assert results[0] == results[1]


def test_manifest_hashes_and_legacy_immutability(config):
    before = cli.legacy_hashes()
    files, manifest = cli.build_artifacts(config, before)
    assert manifest["legacy_hashes_before"] == manifest["legacy_hashes_after"] == before == cli.legacy_hashes()
    assert manifest["test_status"] == "candidate_not_frozen"
    assert set(files) == {"train.jsonl", "validation.jsonl", "test.jsonl", "audit.json", "manifest.json", "DATA_CARD.md"}
    for name, metadata in manifest["files"].items():
        assert metadata == {"sha256": hashlib.sha256(files[name]).hexdigest(), "bytes": len(files[name])}
    assert manifest["source"]["source_hash"] == digest(manifest["source"]["working_tree_file_hashes"])


def test_cli_refuses_legacy_outside_paths(config, tmp_path):
    for path in [str(tmp_path), "data/scenarios_train.jsonl", "data/research_v1", "results/research"]:
        with pytest.raises(ValueError):
            cli.checked_output(path)
    assert cli.checked_output("data/research_v1_candidate") == ROOT / "data/research_v1_candidate"


def test_cli_atomic_output_refusal_and_readonly_check(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "legacy_hashes", lambda: {"legacy": "unchanged"})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    files = {"manifest.json": b"fixture"}
    monkeypatch.setattr(cli, "build_artifacts", lambda c, before: (files, {"counts": {"train": 1200}}))
    monkeypatch.setattr(sys, "argv", ["generate", "--config", str(config_path)])
    assert cli.main() == 0
    output = tmp_path / "data/research_v1_candidate"
    assert (output / "manifest.json").read_bytes() == b"fixture"
    assert cli.main() == 1
    assert "refusing overwrite" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["generate", "--config", str(config_path), "--check"])
    before = (output / "manifest.json").stat().st_mtime_ns
    assert cli.main() == 0
    assert (output / "manifest.json").stat().st_mtime_ns == before
    (output / "manifest.json").write_bytes(b"corrupt")
    assert cli.main() == 1
    assert "byte mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["bool_population", "zero_area", "bad_humidity", "bad_birth_unit", "infinite_temperature", "extra_record_field"])
def test_invalid_snapshots_are_rejected(config, mutation):
    world = make_instance(config, "simple_weather", "base", 0)["world"]
    entity = entity_name(config, 0)
    if mutation == "bool_population":
        world["countries"][entity]["population"] = True
    elif mutation == "zero_area":
        world["countries"][entity]["area_km2"] = 0
    elif mutation == "bad_humidity":
        world["weather"][entity]["humidity"] = 101
    elif mutation == "bad_birth_unit":
        world["search"][entity]["birth_year"]["unit"] = "km"
    elif mutation == "infinite_temperature":
        world["weather"][entity]["temperature"] = float("inf")
    else:
        world["weather"][entity]["extra"] = 1
    with pytest.raises(ValueError):
        ResearchEnvironment(world)


@pytest.mark.parametrize("content", ['{"seed":1,"seed":2}', '{"value":NaN}', '{"value":Infinity}', '[]', '{broken'])
def test_json_loader_fails_closed(tmp_path, content):
    path = tmp_path / "input.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_json(path)


def test_second_generation_seed_passes_all_split_policies(config):
    other = copy.deepcopy(config)
    other["seed"] += 1
    data, _ = generate_dataset(other)
    report = audit_dataset(data, other)
    assert report["status"] == "pass" and report["replayed_references"] == 1620


def test_observation_exposure_is_reported_without_calling_it_label_leakage(audit):
    exposure = audit["train_observation_fact_exposure"]
    for split in ("validation", "test"):
        assert exposure[split]["parameter"]["rows_with_any_fact_previously_observed_in_train"] == 0
        assert exposure[split]["parameter"]["rows_with_all_facts_previously_observed_in_train"] == 0
        assert exposure[split]["iid"]["rows_with_any_fact_previously_observed_in_train"] > 0
    assert all(not any(pair.values()) for pair in audit["cross_split_overlap"].values())


def test_entity_alias_pool_is_unique_and_not_a_visible_numeric_index(config):
    aliases = [entity_name(config, i) for i in range(2048)]
    assert len(set(aliases)) == 2048
    assert all(len(alias.removeprefix("Fiction-")) == 16 for alias in aliases)
    assert aliases[0] != "Fiction-0000" and aliases[1023] != "Fiction-1023"
    assert entity_name(config, 7) == make_instance(config, "simple_weather", "base", 7)["parameters"]["entity"]
    assert entity_name(config, 7) == make_instance(config, "simple_lookup", "test_ood", 7)["parameters"]["entity"]


def test_all_conversion_pairs_in_train(generated):
    pairs = {(r["parameters"]["from_unit"], r["parameters"]["to_unit"]) for r in generated[0]["train"] if r["recipe_id"] == "simple_convert"}
    assert pairs == set(CONVERSION_PAIRS)
    assert len(TOOL_SCHEMAS) == 5
    validate_config(load_json(ROOT / "configs/research/dataset_v1.json"))

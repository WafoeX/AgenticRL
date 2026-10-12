"""R1.6/R2 adversarial CPU tests; no model inference or test-split data reads."""

import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from scripts import diagnose_research as cli
from src.research_answers import (
    MAX_ANSWER_BYTES,
    answer_specification,
    check_answer,
    final_content,
    independent_conversion,
    oracle_answer,
    parse_answer,
)
from src.research_checker import (
    check_attempt,
    check_messages,
    check_trace,
    expression_signature,
)
from src.research_environment import ResearchEnvironment, canonical_json
from src.research_generator import RECIPE_TOOLS, make_instance, render_record
from src.research_interventions import (
    DYNAMIC_RECIPES,
    build_pairs,
    check_pair,
    make_pair,
    summarize_pairs,
)
from src.research_schema import digest, load_json, model_messages

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_json(ROOT / "configs/research/dataset_v1.json")


def fixture_record(config, recipe="country_density_convert", index=13):
    # Fresh synthetic development fixture, NOT an official holdout row.
    return render_record(config, make_instance(config, recipe, "base", index),
                         {"split": "validation", "generalization": "iid", "wording": "train"})


@pytest.mark.parametrize("recipe", sorted(RECIPE_TOOLS))
def test_every_recipe_independent_oracle_and_reference(config, recipe):
    for index in (0, 13, 257, 1023):
        row = fixture_record(config, recipe, index)
        attempt = cli.scripted_attempt(row)
        checked = check_attempt(row, attempt["content"], attempt["trace"])
        assert checked["answer"]["correct"]
        assert checked["trace"]["strict_complete"]
        assert check_answer(row, canonical_json(row["oracle_answer"]))["correct"]


@pytest.mark.parametrize("content", [
    "", "42", "[]", "null", "true", '{"value":true,"unit":"years"}',
    '{"value":"42","unit":"years"}', '{"value":[],"unit":"years"}',
    '{"value":null,"unit":"years"}', '{"value":1,"unit":"years","success":true}',
    '{"value":1,"value":2,"unit":"years"}', '{"value":1,"unit":"years","unit":"years"}',
    '{"value":NaN,"unit":"years"}', '{"value":Infinity,"unit":"years"}',
    '{"value":1e9999,"unit":"years"}', '{"value":1,"unit":"Years"}',
    '{"value":1,"unit":null}', '{"value":1}', '{"value":1,"unit":"years"} {"value":2,"unit":"years"}',
    'Answer: {"value":1,"unit":"years"}', '```json\n{"value":1,"unit":"years"}\n```',
    '{"value":"","unit":"entity"}', '{"value":1,"unit":"entity"}',
    '{"value":{"value":1,"value":2},"unit":"entity"}', '{"value":1,"unit":"years",}',
    " " * (MAX_ANSWER_BYTES + 1), "[" * 2000,
])
def test_strict_parser_rejects_conflicts_types_and_extraction(config, content):
    with pytest.raises(ValueError):
        parse_answer(content)
    assert check_answer(fixture_record(config, "age_calculation"), content)["reason"] == "parse_error"


def test_parser_whitespace_finite_decimal_and_no_enum_aliases(config):
    assert parse_answer(' \n {"unit":"years", "value":2.00} \t') == {"value": Decimal(2), "unit": "years"}
    assert parse_answer('{"value":"sunny","unit":"condition"}')["value"] == "sunny"
    row = fixture_record(config, "weather_compare")
    assert check_answer(row, canonical_json(oracle_answer(row)))["correct"]
    gold = oracle_answer(row)
    assert not check_answer(row, canonical_json({**gold, "value": gold["value"].upper()}))["correct"]
    assert not check_answer(row, '{"value":"B","unit":"entity"}')["correct"]


@pytest.mark.parametrize("source,target,value,expected", [
    ("celsius", "fahrenheit", 0, "32"), ("fahrenheit", "celsius", 212, "100"),
    ("km", "miles", "1.609344", "1"), ("miles", "km", 1, "1.609344"),
    ("kg", "lbs", "0.45359237", "1"), ("lbs", "kg", 1, "0.453592"),
    ("meters", "feet", "0.3048", "1"), ("feet", "meters", 1, "0.3048"),
    ("people_per_km2", "people_per_mile2", 1, "2.589988"),
    ("people_per_mile2", "people_per_km2", "2.589988110336", "1"),
])
def test_independent_conversion_hand_values(source, target, value, expected):
    assert independent_conversion(Decimal(str(value)), source, target) == Decimal(expected)


def test_hand_oracles_and_squared_density_unit_error(config):
    row = fixture_record(config)
    entity = row["parameters"]["entity"]
    row["environment_snapshot"]["countries"][entity].update(population=1000, area_km2=10)
    assert oracle_answer(row) == {"value": 258.998811, "unit": "people_per_mile2"}
    assert not check_answer(row, '{"value":62.137119,"unit":"people_per_mile2"}')["correct"]
    assert check_answer(row, '{"value":258.998811,"unit":"people_per_km2"}')["reason"] == "unit_error"
    age = fixture_record(config, "age_calculation")
    age["parameters"]["year"] = 2026
    age["environment_snapshot"]["search"][age["parameters"]["entity"]]["birth_year"]["value"] = 1990
    assert oracle_answer(age) == {"value": 36, "unit": "years"}


@pytest.mark.parametrize("recipe,table,field", [("weather_compare", "weather", "temperature"), ("country_compare", "countries", "population")])
def test_independent_tie(config, recipe, table, field):
    row = fixture_record(config, recipe)
    p = row["parameters"]
    row["environment_snapshot"][table][p["entity_b"]][field] = row["environment_snapshot"][table][p["entity"]][field]
    assert oracle_answer(row) == {"value": "tie", "unit": "entity"}
    assert check_answer(row, '{"value":"tie","unit":"entity"}')["correct"]
    assert not check_answer(row, '{"value":"Tie","unit":"entity"}')["correct"]


def test_fixed_tolerance_not_mutable_by_record(config):
    row = fixture_record(config, "simple_math")
    gold = oracle_answer(row)
    assert not check_answer(row, canonical_json({**gold, "value": gold["value"] + 1e-5}))["correct"]
    row["answer_spec"].update(abs_tol=1e20, rel_tol=1e20)
    assert not check_answer(row, canonical_json({**gold, "value": gold["value"] + 1}))["correct"]
    row = fixture_record(config, "population_math")
    gold = oracle_answer(row)
    tolerance = max(1e-6, 1e-9 * abs(gold["value"]))
    assert check_answer(row, canonical_json({**gold, "value": gold["value"] + tolerance * .5}))["correct"]
    assert not check_answer(row, canonical_json({**gold, "value": gold["value"] + tolerance * 2}))["correct"]
    assert answer_specification(row)["abs_tol"] == 1e-6


def test_oracle_ignores_all_saved_golds_reward_reference_and_executor(config, monkeypatch):
    row = fixture_record(config)
    gold = oracle_answer(row)
    for key in ("oracle_answer", "reference_answer", "reference_trace", "expected_calls", "dependency_graph", "answer_spec",
                "semantic_payload", "value_payload", "reference_depth", "dynamic_dependency"):
        row[key] = {"success": True, "reward": 999, "value": -777}
    monkeypatch.setattr(ResearchEnvironment, "execute", lambda *args: "Error: broken tool implementation")
    assert oracle_answer(row) == gold
    assert check_answer(row, canonical_json(gold))["correct"]
    assert not check_answer(row, '{"value":-777,"unit":"people_per_mile2"}')["correct"]
    # An actual fact change DOES affect the independent oracle.
    row["environment_snapshot"]["countries"][row["parameters"]["entity"]]["population"] += 1000
    assert oracle_answer(row) != gold


def test_final_answer_selects_terminal_assistant_only(config):
    content = canonical_json(oracle_answer(fixture_record(config)))
    assert final_content([{"role": "assistant", "content": content}]) == content
    for messages in ([], [{"role": "assistant", "content": content}, {"role": "tool", "content": "1"}],
                     [{"role": "assistant", "content": content, "tool_calls": [{"id": "x"}]}],
                     [{"role": "assistant", "content": None}]):
        with pytest.raises(ValueError):
            final_content(messages)
    assert final_content([{"role": "assistant", "content": content}, {"role": "assistant", "content": "wrong"}]) == "wrong"


@pytest.mark.parametrize("mutation,error", [
    ("missing", "missing_step"), ("wrong_args", "wrong_arguments_or_route"), ("forged", "fabricated_observation"),
    ("same_turn", "same_turn_dependency"), ("reversed", "wrong_arguments_or_route"),
    ("tool_error", "tool_execution_error"), ("bool_turn", "invalid_trace"), ("negative_turn", "invalid_trace"),
    ("extra_field", "invalid_trace"), ("non_string_observation", "invalid_trace"),
])
def test_adversarial_trace_answer_not_reward_or_protocol_gated(config, mutation, error):
    row = fixture_record(config)
    attempt = cli.scripted_attempt(row)
    trace = attempt["trace"]
    if mutation == "missing":
        trace.pop()
    elif mutation == "wrong_args":
        trace[2]["arguments"]["value"] += 100
        trace[2]["observation"] = ResearchEnvironment(row["environment_snapshot"]).execute(trace[2]["name"], trace[2]["arguments"])
    elif mutation == "forged":
        trace[0]["observation"] = '{}'
    elif mutation == "same_turn":
        trace[1]["turn"] = 0
    elif mutation == "reversed":
        trace.reverse()
        for i, step in enumerate(trace):
            step["turn"] = i
    elif mutation == "tool_error":
        trace[2]["arguments"]["value"] = True
    elif mutation == "bool_turn":
        trace[0]["turn"] = True
    elif mutation == "negative_turn":
        trace[0]["turn"] = -1
    elif mutation == "extra_field":
        trace[0]["success"] = True
    else:
        trace[0]["observation"] = {}
    checked = check_attempt(row, attempt["content"], trace)
    assert checked["answer"]["correct"]  # even missing-step/high-reward-looking routes don't gate answer
    assert not checked["trace"]["strict_complete"]
    assert error in checked["trace"]["errors"]


def test_no_tools_correct_answer_and_complete_protocol_wrong_answer(config):
    row = fixture_record(config)
    attempt = cli.scripted_attempt(row)
    direct = check_attempt(row, attempt["content"], [])
    assert direct["answer"]["correct"] and not direct["trace"]["strict_complete"]
    assert not direct["trace"]["semantic_valid"]  # no trace evidence, not proof of a tool route
    wrong = check_attempt(row, '{"value":-999,"unit":"people_per_mile2"}', attempt["trace"])
    assert wrong["trace"]["strict_complete"] and not wrong["answer"]["correct"]
    assert check_trace(row, None)["errors"] == ["invalid_trace"]
    assert check_trace(row, attempt["trace"] * 30)["errors"] == ["invalid_trace"]


@pytest.mark.parametrize("status", ["truncated", "exception", "infra_failure", "missing_output"])
def test_termination_is_explicit_and_no_success(config, status):
    row = fixture_record(config)
    attempt = cli.scripted_attempt(row)
    checked = check_attempt(row, attempt["content"], attempt["trace"], termination=status)
    assert not checked["answer"]["correct"] and not checked["trace"]["strict_complete"]
    assert checked["answer"]["reason"] == status
    assert status in checked["trace"]["errors"]


@pytest.mark.parametrize("recipe", ["weather_compare", "country_compare"])
def test_independent_queries_reordering_same_turn_allowed(config, recipe):
    row = fixture_record(config, recipe)
    trace = cli.scripted_attempt(row)["trace"][::-1]
    for step in trace:
        step["turn"] = 0
    assert check_trace(row, trace)["strict_complete"]
    assert check_trace(row, trace)["dynamic_status"] == "not_applicable"


def test_commutative_expressions_not_numeric_coincidence(config):
    row = fixture_record(config, "simple_math", 0)
    trace = cli.scripted_attempt(row)["trace"]
    a, b = row["parameters"]["a"], row["parameters"]["b"]
    trace[0]["arguments"]["expression"] = f"({b}.0) + (+{a})"
    trace[0]["observation"] = ResearchEnvironment(row["environment_snapshot"]).execute("calculator", trace[0]["arguments"])
    assert check_trace(row, trace)["strict_complete"]
    assert expression_signature("2*2") != expression_signature("2+2")
    assert expression_signature("3-4") != expression_signature("4-3")
    trace[0]["arguments"]["expression"] = str(a + b)
    trace[0]["observation"] = canonical_json(a + b)
    assert not check_trace(row, trace)["semantic_valid"]


def test_extra_truthful_calls_semantic_route_but_not_strict(config):
    row = fixture_record(config)
    trace = cli.scripted_attempt(row)["trace"]
    trace.append({"turn": 3, "name": "calculator", "arguments": {"expression": "1+1"}, "observation": "2.0"})
    result = check_trace(row, trace)
    assert result["semantic_valid"] and result["dynamic_status"] == "valid" and not result["strict_complete"]
    assert "extra_calls" in result["errors"]
    trace[-1]["observation"] = "3.0"
    assert not check_trace(row, trace)["semantic_valid"]


@pytest.mark.parametrize("recipe", sorted(DYNAMIC_RECIPES))
def test_pairs_really_perturb_consumers_and_answer_no_prompt_changes(config, recipe):
    row = fixture_record(config, recipe)
    before = copy.deepcopy(row)
    pair = make_pair(row)
    a, b = cli.scripted_attempt(pair["base"]), cli.scripted_attempt(pair["variant"])
    result = check_pair(pair, a, b)
    assert result["responsive"] and result["consumers_changed"] and result["dependency_valid"]
    assert pair["group_id"] == row["semantic_group_id"]
    assert pair["base"]["task"] == pair["variant"]["task"]
    assert model_messages(pair["base"]) == model_messages(pair["variant"]) == model_messages(row)
    assert pair["variant"]["environment_hash"] == digest(pair["variant"]["environment_snapshot"])
    assert "oracle_answer" not in pair["variant"] and "reference_trace" not in pair["variant"]
    assert row == before
    env_a, env_b = (ResearchEnvironment(pair[side]["environment_snapshot"]) for side in ("base", "variant"))
    first = a["trace"][0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(lambda i: (env_a if i % 2 == 0 else env_b).execute(first["name"], first["arguments"]), range(20)))
    assert observations == [a["trace"][0]["observation"] if i % 2 == 0 else b["trace"][0]["observation"] for i in range(20)]
    frozen = cli.frozen_attempt(pair["variant"], a)
    frozen_result = check_pair(pair, a, frozen)
    assert frozen_result["variant"]["trace"]["execution_valid"]
    assert not frozen_result["responsive"] and not frozen_result["consumers_changed"]
    stale = copy.deepcopy(b)
    stale["content"] = a["content"]
    assert not check_pair(pair, a, stale)["responsive"]
    assert check_pair(pair, a, stale)["variant"]["trace"]["strict_complete"]
    assert not check_pair(pair, a, a)["variant"]["trace"]["execution_valid"]


def test_one_call_fahrenheit_alternative_is_answer_correct_edge_bypass(config):
    pair = make_pair(fixture_record(config, "weather_convert"))
    attempts = []
    for side in ("base", "variant"):
        view = pair[side]
        args = {"city": view["parameters"]["entity"], "units": "fahrenheit"}
        observation = ResearchEnvironment(view["environment_snapshot"]).execute("get_weather", args)
        content = canonical_json({"value": json.loads(observation)["temperature"], "unit": "fahrenheit"})
        attempts.append({"content": content, "trace": [{"turn": 0, "name": "get_weather", "arguments": args, "observation": observation}]})
    result = check_pair(pair, *attempts)
    assert result["responsive"] and result["alternative_bypass"]
    assert not result["dependency_valid"] and not result["consumers_changed"]
    for side in ("base", "variant"):
        assert result[side]["answer"]["correct"] and result[side]["trace"]["semantic_valid"]
        assert not result[side]["trace"]["strict_complete"]


def test_pair_guard_test_duplicates_non_dynamic_and_denominators(config):
    row = fixture_record(config)
    pair = make_pair(row)
    with pytest.raises(ValueError):
        build_pairs([row, row])
    test_row = copy.deepcopy(row)
    test_row["split"] = "test"
    with pytest.raises(ValueError):
        make_pair(test_row)
    with pytest.raises(ValueError):
        build_pairs([test_row])
    simple = fixture_record(config, "simple_math")
    with pytest.raises(ValueError):
        make_pair(simple)
    assert build_pairs([simple]) == []
    summary = summarize_pairs([pair], [])
    assert summary["planned_groups"] == summary["missing_groups"] == 1
    assert summary["responsive_rate"] == 0
    assert summarize_pairs([], [])["responsive_rate"] is None
    a, b = cli.scripted_attempt(pair["base"]), cli.scripted_attempt(pair["variant"])
    checked = check_pair(pair, a, b)
    with pytest.raises(ValueError):
        summarize_pairs([pair], [checked, checked])
    with pytest.raises(ValueError):
        summarize_pairs([pair], [{**checked, "group_id": "unknown"}])


def test_full_validation_artifacts_and_controls_reproduce():
    files = cli.build_artifacts()
    assert files == cli.build_artifacts()
    summary = json.loads(files["summary.json"])
    assert summary["references"] == {"planned": 180, "answer_correct": 180, "strict_complete": 180}
    assert summary["paired"]["planned_groups"] == summary["paired"]["responsive_groups"] == 106
    assert summary["paired"]["consumer_changed_groups"] == 106
    assert summary["legal_fahrenheit_alternative_groups"] > 0
    assert summary["test_used"] is False and summary["test_frozen"] is False
    assert all(value == {"planned_groups": 106, "rejected_groups": 106} for value in summary["negative_controls"].values())
    for name, info in json.loads(files["manifest.json"])["files"].items():
        import hashlib
        assert hashlib.sha256(files[name]).hexdigest() == info["sha256"]
        assert len(files[name]) == info["bytes"]


def test_cli_refuses_overwrite_forbidden_outputs_and_read_only_check(tmp_path):
    # Fixed input; output-path policy tested without a GPU or arbitrary datasets.
    for output in (str(tmp_path), "data/research_v1_candidate", "results/research"):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_research.py"), "--output", output],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        assert result.returncode == 1 and "output must" in result.stderr


def test_terminal_message_adapter_never_selects_earlier_success(config):
    row = fixture_record(config)
    attempt = cli.scripted_attempt(row)
    correct = {"role": "assistant", "content": attempt["content"]}
    assert check_messages(row, [correct], attempt["trace"])["answer"]["correct"]
    for last in ({"role": "tool", "content": "1"}, {**correct, "tool_calls": [{"id": "x"}]},
                 {"role": "assistant", "content": None}):
        checked = check_messages(row, [correct, last], attempt["trace"])
        assert not checked["answer"]["correct"] and checked["answer"]["reason"] == "missing_output"
    assert not check_messages(row, [correct, {"role": "assistant", "content": "wrong"}], attempt["trace"])["answer"]["correct"]
    assert check_messages(row, [], [], termination="infra_failure")["answer"]["reason"] == "infra_failure"


def test_fractional_facts_not_silently_truncated_and_trace_metadata_not_trusted(config):
    row = fixture_record(config, "simple_lookup")
    entity = row["parameters"]["entity"]
    row["environment_snapshot"]["countries"][entity]["gdp_billion_usd"] = 1.5
    assert oracle_answer(row) == {"value": 1.5, "unit": "billion_usd"}
    assert check_answer(row, '{"value":1.5,"unit":"billion_usd"}')["correct"]
    row = fixture_record(config)
    attempt = cli.scripted_attempt(row)
    for key in ("expected_calls", "reference_trace", "dependency_graph", "dynamic_dependency", "reference_depth"):
        row[key] = "POISONED"
    assert check_attempt(row, attempt["content"], attempt["trace"])["trace"]["strict_complete"]


def test_cli_atomic_output_check_and_refusal_are_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_artifacts", lambda: {"fixture.json": b'{"ok":true}\n'})
    monkeypatch.setattr(sys, "argv", ["diagnose_research"])
    assert cli.main() == 0
    output = tmp_path / "results/research/checker_interventions_v1"
    original = (output / "fixture.json").read_bytes()
    assert cli.main() == 1
    assert (output / "fixture.json").read_bytes() == original
    monkeypatch.setattr(sys, "argv", ["diagnose_research", "--check"])
    assert cli.main() == 0
    assert (output / "fixture.json").read_bytes() == original
    (output / "extra").write_text("extra", encoding="utf-8")
    assert cli.main() == 1
    assert (output / "extra").exists()

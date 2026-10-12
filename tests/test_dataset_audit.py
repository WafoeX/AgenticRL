"""CPU-only regression tests for the read-only legacy audit."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.dataset_audit import (
    audit_repository,
    finite_scenarios,
    normalized_task,
    pair_overlap,
    replay_reference,
    semantic_probes,
    summarize_rows,
    template_capacities,
)
from src.scenarios import SCENARIO_TEMPLATES, generate_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    return audit_repository(ROOT)


def test_current_capacities():
    assert template_capacities() == {
        "age_calculation": 4, "weather_convert": 5, "population_math": 5,
        "simple_math": 291060, "simple_weather": 7, "simple_search": 5,
        "search_and_convert": 1, "country_density_convert": 4,
        "weather_compare": 9, "simple_convert": 100, "simple_lookup": 5,
        "country_compare": 4,
    }


def test_exhaustive_choice_enumeration():
    template = {
        "id": "example", "task": "{a}:{b}", "expected_tools": ["calculator"],
        "params": lambda rng: {"a": rng.choice(["x", "y"]), "b": rng.randint(1, 3)},
        "build_calls": lambda p: [{"name": "calculator", "arguments": {"expression": str(p["b"])}}],
    }
    assert len(finite_scenarios(template)) == 6
    assert [row["task"] for row in finite_scenarios(template)] == ["x:1", "x:2", "x:3", "y:1", "y:2", "y:3"]


def test_large_domain_rejected():
    math_template = next(t for t in SCENARIO_TEMPLATES if t["id"] == "simple_math")
    with pytest.raises(ValueError, match="finite-audit limit"):
        finite_scenarios(math_template)


def test_math_capacity_fails_closed_on_template_drift(monkeypatch):
    math_template = next(t for t in SCENARIO_TEMPLATES if t["id"] == "simple_math")
    monkeypatch.setitem(math_template, "task", "What is {a}?")
    with pytest.raises(ValueError, match="capacity assumptions changed"):
        template_capacities()


def test_dataset_counts_and_overlap(report):
    assert (report["datasets"]["train"]["rows"], report["datasets"]["train"]["unique_tasks"]) == (200, 86)
    assert (report["datasets"]["validation"]["rows"], report["datasets"]["validation"]["unique_tasks"]) == (50, 37)
    overlap = report["overlap"]["train_to_validation"]
    assert overlap["right_rows_with_exact_text_in_left"] == 36
    assert overlap["shared_exact_texts"] == 24
    assert report["legacy_test_file_exists"] is False


def test_normalized_overlap_does_not_use_only_exact_match():
    row = generate_scenario(42)
    alternate = copy.deepcopy(row)
    alternate["task"] = "  " + row["task"].upper().rstrip(".?!") + "  "
    result = pair_overlap([row], [alternate])
    assert result["right_rows_with_exact_text_in_left"] == 0
    assert result["right_rows_with_normalized_text_in_left"] == 1
    assert normalized_task(" X   y? ") == "x y"


def test_unknown_family_is_not_silently_dropped():
    row = generate_scenario(42)
    row["id"] = "new_unrecognized_family"
    with pytest.raises(ValueError, match="Unknown legacy template"):
        summarize_rows([row])


def test_missing_calls_and_sequence_mismatch_rejected():
    row = generate_scenario(42)
    row["expected_calls"] = []
    with pytest.raises(ValueError, match="Missing expected calls"):
        summarize_rows([row])
    row = generate_scenario(42)
    row["expected_tools"] = ["wrong_tool"]
    with pytest.raises(ValueError, match="sequence differs"):
        summarize_rows([row])


def test_missing_input_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Cannot read audit input"):
        audit_repository(tmp_path)


def test_invalid_json_fails_closed(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/scenarios_train.jsonl").write_text("{invalid json\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        audit_repository(tmp_path)


def test_guido_observation_does_not_support_hidden_label(report):
    probes = report["semantic_probes"]["age_search_coverage"]
    guido = [probe for probe in probes if "Guido" in probe["task"]]
    einstein = [probe for probe in probes if "Einstein" in probe["task"]]
    assert len(guido) == len(einstein) == 2
    assert all(not probe["observation_contains_reference_birth_year"] for probe in guido)
    assert all(probe["observation_contains_reference_birth_year"] for probe in einstein)
    assert all("Einstein" in probe["search_result"][0]["text"] for probe in guido)


def test_comparison_position_shortcuts(report):
    assert report["semantic_probes"]["comparison_winner_slots"] == {
        "weather_compare": {"slot_b": 9}, "country_compare": {"slot_b": 4},
    }


def test_density_executes_but_is_semantically_wrong(report):
    probes = report["semantic_probes"]["density_units"]
    assert len(probes) == 4
    assert all(0.75 < probe["relative_error"] < 0.77 for probe in probes)
    assert all(probe["legacy_reference_value"] < probe["independent_people_per_square_mile"] for probe in probes)
    assert all(probe["trace"][-1]["arguments"]["from_unit"] == "km" for probe in probes)


def test_weather_has_valid_one_call_alternative(report):
    assert len(report["semantic_probes"]["weather_alternative_solution"]) == 5
    assert all(probe["one_call_value"] == probe["two_call_value"]
               for probe in report["semantic_probes"]["weather_alternative_solution"])


def test_static_search_convert_does_not_consume_observation(report):
    trace = report["semantic_probes"]["search_convert_static_reference"]
    assert trace[1]["arguments"]["value"] == 149600000
    assert trace[1]["arguments"]["from_unit"] == "km"


def test_reference_dynamic_resolution():
    template = next(t for t in SCENARIO_TEMPLATES if t["id"] == "weather_convert")
    trace = replay_reference(finite_scenarios(template)[0])
    observed = json.loads(trace[0]["result"])
    assert trace[1]["arguments"]["value"] == observed["temperature"]
    assert [entry["turn"] for entry in trace] == [1, 2]


def test_unsupported_placeholder_fails_closed():
    scenario = {"expected_calls": [
        {"name": "calculator", "arguments": {"expression": "__dynamic__"}},
    ]}
    with pytest.raises(ValueError, match="no preceding observation"):
        replay_reference(scenario)


def test_invalid_tool_observation_fails_closed(monkeypatch):
    monkeypatch.setattr("src.dataset_audit.execute_tool", lambda *_: "Error: invalid output")
    with pytest.raises(ValueError, match="Invalid JSON"):
        semantic_probes()


def test_inputs_unchanged_and_report_deterministic():
    paths = [ROOT / "data/scenarios_train.jsonl", ROOT / "data/scenarios_val.jsonl",
             ROOT / "src/tools.py", ROOT / "src/scenarios.py"]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert audit_repository(ROOT) == audit_repository(ROOT)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


def test_cli_default_is_read_only():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_legacy_dataset.py")],
                            cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Read-only summary; no report file written." in result.stdout


def test_cli_refuses_dataset_output_path():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_legacy_dataset.py"),
                             "--output", str(ROOT / "data/scenarios_train.jsonl"), "--overwrite"],
                            cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "output must be a file under" in result.stderr

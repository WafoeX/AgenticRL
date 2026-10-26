"""Read-only audit of the legacy benchmark, independent of ART and rewards.

This is NOT a final-answer oracle or a new dataset generator. Capacity counts
refer to raw legacy task strings, not independent semantic examples. Semantic
findings are explicit probes of the currently versioned local environment.
"""

import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.scenarios import SCENARIO_TEMPLATES
from src.tools import execute_tool

AUDIT_VERSION = "legacy-template-audit-v1"


def _decode_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid JSON in audit input or tool observation") from exc


def _numeric_observation(text):
    value = _decode_json(text)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric tool observation")
    return value


def _load_rows(path):
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(f"Cannot read audit input: {path}") from exc
    return [_decode_json(line) for line in text.splitlines() if line.strip()]


class _NeedChoice(Exception):
    def __init__(self, values):
        self.values = list(values)
        super().__init__("Unassigned random choice")


class _BranchRandom:
    """Enumerate small finite domains by replaying the generator's RNG calls."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.cursor = 0

    def choice(self, values):
        if self.cursor == len(self.prefix):
            raise _NeedChoice(values)
        value = self.prefix[self.cursor]
        self.cursor += 1
        if value not in values:
            raise ValueError("Generator choices changed between branches")
        return value

    def randint(self, low, high):
        return self.choice(range(low, high + 1))


def finite_scenarios(template, *, max_branches=2000):
    """Exhaust all small choice domains; deliberately reject large domains."""
    pending: list[tuple[Any, ...]] = [()]
    scenarios = []
    visited = 0
    while pending:
        prefix = pending.pop()
        visited += 1
        if visited > max_branches:
            raise ValueError("Domain exceeds finite-audit limit")
        try:
            params = template["params"](_BranchRandom(prefix))
        except _NeedChoice as branch:
            if len(branch.values) + len(pending) > max_branches:
                raise ValueError("Domain exceeds finite-audit limit") from branch
            pending.extend((*prefix, value) for value in branch.values)
            continue
        public = {key: value for key, value in params.items() if not key.startswith("_")}
        scenarios.append({
            "id": template["id"],
            "task": template["task"].format(**public),
            "expected_tools": template["expected_tools"],
            "expected_calls": template["build_calls"](params),
        })
    return sorted(scenarios, key=lambda row: row["task"])


class _MathDomainRandom:
    def __init__(self):
        self.domains = []

    def choice(self, values):
        self.domains.append(list(values))
        return values[0]

    def randint(self, low, high):
        self.domains.append(range(low, high + 1))
        return low


def template_capacities():
    """Exact raw string counts for current templates, guarded math formula."""
    results = {}
    for template in SCENARIO_TEMPLATES:
        if template["id"] == "simple_math":
            rng = _MathDomainRandom()
            template["params"](rng)
            # The current rendering is injective in (op, a, b). Fail on drift
            # rather than claiming this analytical formula for an unknown form.
            expected_task = "What is {a} {op_word} {b}?"
            expected_domains = [["+", "-", "*"], range(10, 1000), range(2, 100)]
            if template["task"] != expected_task or rng.domains != expected_domains:
                raise ValueError("simple_math capacity assumptions changed")
            results[template["id"]] = math.prod(len(domain) for domain in rng.domains)
        else:
            rows = finite_scenarios(template)
            results[template["id"]] = len({row["task"] for row in rows})
    return results


def normalized_task(task):
    return " ".join(task.casefold().split()).rstrip(".?!")


def protocol_signature(row):
    """Diagnostic only: shared placeholder protocol is NOT semantic identity."""
    return json.dumps(row["expected_calls"], sort_keys=True, separators=(",", ":"))


def _validate_row(row):
    families = {template["id"] for template in SCENARIO_TEMPLATES}
    if not isinstance(row, dict) or row.get("id") not in families:
        raise ValueError("Unknown legacy template")
    if not isinstance(row.get("task"), str) or not row["task"].strip():
        raise ValueError("Missing task text")
    calls = row.get("expected_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("Missing expected calls")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("Malformed expected call")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError("Malformed arguments")
    if row.get("expected_tools") != [call["name"] for call in calls]:
        raise ValueError("Expected tool sequence differs from calls")


def summarize_rows(rows):
    for row in rows:
        _validate_row(row)
    families = {}
    for family in sorted({row["id"] for row in rows}):
        subset = [row for row in rows if row["id"] == family]
        families[family] = {
            "rows": len(subset),
            "unique_tasks": len({row["task"] for row in subset}),
            "unique_protocol_signatures": len({protocol_signature(row) for row in subset}),
        }
    return {
        "rows": len(rows),
        "unique_tasks": len({row["task"] for row in rows}),
        "unique_normalized_tasks": len({normalized_task(row["task"]) for row in rows}),
        "depth_counts": dict(sorted(Counter(len(row["expected_calls"]) for row in rows).items())),
        "families": families,
    }


def pair_overlap(left, right):
    left_texts = {row["task"] for row in left}
    left_normalized = {normalized_task(row["task"]) for row in left}
    left_protocols = {protocol_signature(row) for row in left}
    return {
        "right_rows_with_exact_text_in_left": sum(row["task"] in left_texts for row in right),
        "shared_exact_texts": len(left_texts & {row["task"] for row in right}),
        "right_rows_with_normalized_text_in_left": sum(
            normalized_task(row["task"]) in left_normalized for row in right
        ),
        "right_rows_with_protocol_signature_in_left": sum(
            protocol_signature(row) in left_protocols for row in right
        ),
        "note": "Placeholder protocol overlap is diagnostic, not a semantic leakage count.",
    }


def replay_reference(scenario):
    """Concretize supported legacy placeholders and execute a reference trace.

    Results demonstrate execution support, NOT semantic correctness. In
    particular, the density template executes successfully with wrong units.
    """
    trace = []
    for turn, call in enumerate(scenario["expected_calls"], start=1):
        arguments = dict(call["arguments"])
        for key, value in arguments.items():
            if value != "__dynamic__":
                continue
            if not trace:
                raise ValueError("Dynamic call has no preceding observation")
            previous = _decode_json(trace[-1]["result"])
            if key == "expression" and trace[-1]["name"] == "lookup_db":
                arguments[key] = f"{previous['population']}/{previous['area_km2']}"
            elif key == "value" and trace[-1]["name"] == "get_weather":
                arguments[key] = previous["temperature"]
            elif key == "value" and isinstance(previous, (int, float)):
                arguments[key] = previous
            else:
                raise ValueError("Unsupported dynamic placeholder")
        result = execute_tool(call["name"], arguments)
        trace.append({"name": call["name"], "arguments": arguments, "result": result, "turn": turn})
    return trace


def semantic_probes():
    templates = {template["id"]: template for template in SCENARIO_TEMPLATES}
    age = []
    for row in finite_scenarios(templates["age_calculation"]):
        trace = replay_reference(row)
        query = trace[0]["arguments"]["query"]
        snippets = _decode_json(trace[0]["result"])
        born = re.findall(r"born[^.]*\b(\d{4})\b", " ".join(item["text"] for item in snippets))
        expected_birth = trace[1]["arguments"]["expression"].split("-")[-1]
        age.append({
            "task": row["task"], "query": query, "search_result": snippets,
            "reference_birth_year": expected_birth,
            "observation_contains_reference_birth_year": expected_birth in born,
        })
    comparisons = {}
    for family, field in (("weather_compare", "temperature"), ("country_compare", "population")):
        counts = Counter()
        for row in finite_scenarios(templates[family]):
            trace = replay_reference(row)
            values = [_decode_json(entry["result"])[field] for entry in trace]
            counts["slot_a" if values[0] > values[1] else "slot_b" if values[1] > values[0] else "tie"] += 1
        comparisons[family] = dict(sorted(counts.items()))
    density = []
    for row in finite_scenarios(templates["country_density_convert"]):
        trace = replay_reference(row)
        record = _decode_json(trace[0]["result"])
        # Independent dimensional formula: 1 international mile = 1.609344 km.
        correct = record["population"] / record["area_km2"] * (1.609344 ** 2)
        actual = _numeric_observation(trace[-1]["result"])
        density.append({
            "task": row["task"], "legacy_reference_value": actual,
            "independent_people_per_square_mile": correct,
            "relative_error": abs(actual - correct) / correct,
            "trace": trace,
        })
    weather_shortcuts = []
    for row in finite_scenarios(templates["weather_convert"]):
        trace = replay_reference(row)
        city = trace[0]["arguments"]["city"]
        direct = _decode_json(execute_tool("get_weather", {"city": city, "units": "fahrenheit"}))
        weather_shortcuts.append({
            "task": row["task"], "two_call_value": _numeric_observation(trace[-1]["result"]),
            "one_call_value": direct["temperature"],
        })
    search_convert = replay_reference(finite_scenarios(templates["search_and_convert"])[0])
    # Representative finite templates and a deterministic math sample. Their
    # successful execution must not be confused with an answer oracle.
    reference_examples = {}
    for family, template in templates.items():
        if family == "simple_math":
            params = template["params"](random.Random(0))
            row = {"id": family, "expected_calls": template["build_calls"](params)}
        else:
            row = finite_scenarios(template)[0]
        reference_examples[family] = replay_reference(row)
    return {
        "age_search_coverage": age,
        "comparison_winner_slots": comparisons,
        "density_units": density,
        "weather_alternative_solution": weather_shortcuts,
        "search_convert_static_reference": search_convert,
        "reference_examples": reference_examples,
        "limits": [
            "Reference execution is not a final-answer checker.",
            "Capacity is raw text capacity; commutative/semantic equivalence can reduce it.",
            "No model inference or GPU training was performed.",
            "Observation matching is not proof of causal consumption.",
        ],
    }


def audit_repository(root: Path):
    paths = {
        "train": root / "data/scenarios_train.jsonl",
        "validation": root / "data/scenarios_val.jsonl",
    }
    datasets = {}
    file_hashes = {}
    for name, path in paths.items():
        datasets[name] = _load_rows(path)
        file_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("src/tools.py", "src/scenarios.py", "src/dataset_audit.py"):
        file_hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    summaries = {name: summarize_rows(rows) for name, rows in datasets.items()}
    capacities = template_capacities()
    for name, summary in summaries.items():
        for family, counts in summary["families"].items():
            counts["raw_task_capacity"] = capacities[family]
            if counts["unique_tasks"] > capacities[family]:
                raise ValueError(f"{name}/{family} exceeds current generator capacity")
    return {
        "audit_version": AUDIT_VERSION,
        "input_hashes": file_hashes,
        "datasets": summaries,
        "overlap": {
            f"{left}_to_{right}": pair_overlap(datasets[left], datasets[right])
            for left, right in itertools.combinations(datasets, 2)
        },
        "legacy_test_file_exists": (root / "data/scenarios_test.jsonl").exists(),
        "raw_task_capacities": capacities,
        "semantic_probes": semantic_probes(),
    }

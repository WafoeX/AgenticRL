"""Server-run reward regressions; these tests never call a model or a judge."""

import copy
from dataclasses import asdict

import pytest

from src.reward_v2 import (
    compute_reward_v2,
    is_complete_tool_chain,
    normalize_expression,
    normalize_value,
    score_final_answer,
)
from src.rewards import compute_reward, compute_reward_by_version, compute_reward_v1


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def trace(calls, results):
    return [{**copy.deepcopy(c), "result": r} for c, r in zip(calls, results, strict=True)]


@pytest.mark.parametrize("predicted,expected,reward", [
    ([call("calculator", expression="3+5")], [call("calculator", expression="3+5")], 4.0),
    ([call("calculator", expression="3 + 5")], [call("calculator", expression="3+5")], 2.0),
    ([call("search", query="wrong")], [call("calculator", expression="3+5")], -2.0),
    ([], [call("calculator", expression="3+5")], -3.0),
    ([call("calculator", expression="3+5")], [], 1.0),
    ([call("convert_units", value=15.0, from_unit="kg", to_unit="lbs")],
     [call("convert_units", value=15, from_unit="kg", to_unit="lbs")], 10 / 3),
    ([call("calculator", expression="3+5"), call("lookup_db", table="countries", key="France")],
     [call("lookup_db", table="countries", key="France"), call("calculator", expression="3+5")], 4.0),
    ([call("get_weather", city="London")],
     [call("get_weather", city="London"), call("get_weather", city="Mumbai")], 3.0),
    ([call("lookup_db", table="countries", key="Brazil")],
     [call("lookup_db", table="countries", key="France"),
      call("lookup_db", table="countries", key="Brazil")], 2.5),
])
def test_v1_frozen_golden_outputs_and_dispatch(predicted, expected, reward):
    assert compute_reward_v1 is compute_reward
    assert compute_reward(predicted, expected) == pytest.approx(reward)
    # v1 must ignore observation shape/content, not reinterpret old experiments.
    assert compute_reward_by_version(
        predicted, expected, version="v1", observations=object(),
    ) == pytest.approx(reward)
    assert compute_reward_by_version(predicted, expected) == compute_reward(predicted, expected)


def test_duplicate_weather_calls_are_not_deduplicated():
    expected = [call("get_weather", city="London"), call("get_weather", city="Mumbai")]
    partial = compute_reward_v2(expected[:1], expected)
    complete = compute_reward_v2(expected, expected)
    assert -3 < partial.total < complete.total
    assert partial.total <= 2
    assert partial.tool_score == 0.5
    assert partial.tool_count_match == 0.5
    assert complete.total == pytest.approx(4)


def test_partial_country_compare_is_not_complete_success():
    expected = [call("lookup_db", table="countries", key="France"),
                call("lookup_db", table="countries", key="Brazil")]
    predicted = expected[1:]
    observations = trace(predicted, [{"population": 216400000}])
    assert compute_reward(predicted, expected) == pytest.approx(2.5)
    assert compute_reward_v2(predicted, expected, observations=observations).total <= 2
    assert not is_complete_tool_chain(predicted, expected, observations=observations)


def test_order_and_incremental_prefix_credit():
    expected = [call("lookup_db", table="countries", key="France"),
                call("calculator", expression="100/2"),
                call("convert_units", value=50, from_unit="km", to_unit="miles")]
    wrong = [expected[1], expected[0], expected[2]]
    full = compute_reward_v2(expected, expected)
    first = compute_reward_v2(expected[:1], expected)
    second = compute_reward_v2(expected[:2], expected)
    assert -3 < first.total < second.total < full.total
    assert full.total > compute_reward_v2(wrong, expected).total
    assert full.exact_sequence_match == 1
    assert full.prefix_match == 1
    assert second.prefix_match == pytest.approx(2 / 3)
    assert second.tool_count_match == pytest.approx(2 / 3)


@pytest.mark.parametrize("value", [15, 15.0, "15"])
def test_dynamic_temperature_uses_previous_observation(value):
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]
    predicted = [expected[0], call("convert_units", value=value, from_unit="celsius", to_unit="fahrenheit")]
    observations = trace(predicted, [{"temperature": 15, "humidity": 80, "units": "celsius"}, "59.0"])
    breakdown = compute_reward_v2(predicted, expected, observations=observations)
    assert breakdown.dynamic_score > 0
    assert breakdown.total == pytest.approx(4)
    assert is_complete_tool_chain(predicted, expected, observations=observations)
    # This standalone fixture supplies a successful result explicitly. The
    # real tool's rejection of numeric strings is checked separately below.


@pytest.mark.parametrize("bad", ["__dynamic__", None, True, float("nan"), float("inf"), 80, 20])
def test_dynamic_missing_invalid_or_unrelated_values_do_not_get_credit(bad):
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]
    predicted = [expected[0], call("convert_units", value=bad, from_unit="celsius", to_unit="fahrenheit")]
    observations = trace(predicted, [{"temperature": 15, "humidity": 80}, "59"])
    breakdown = compute_reward_v2(predicted, expected, observations=observations)
    assert breakdown.dynamic_score == 0
    assert breakdown.total <= 2
    assert not is_complete_tool_chain(predicted, expected, observations=observations)


def test_dynamic_requires_prior_not_future_result_and_trace_binding():
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]
    predicted = [expected[0], call("convert_units", value=15, from_unit="celsius", to_unit="fahrenheit")]
    observations = trace(predicted, [{"temperature": 12}, {"temperature": 15}])
    assert compute_reward_v2(predicted, expected, observations=observations).dynamic_score == 0
    observations = trace(predicted, [{"temperature": 15}, "59"])
    observations[0]["arguments"]["city"] = "Mumbai"
    assert compute_reward_v2(predicted, expected, observations=observations).dynamic_score == 0
    assert not is_complete_tool_chain(predicted, expected, observations=observations)


def test_dynamic_requires_consistent_weather_units():
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]
    predicted = [expected[0], call("convert_units", value=59, from_unit="celsius", to_unit="fahrenheit")]
    observations = trace(predicted, [{"temperature": 59, "units": "fahrenheit"}, "138.2"])
    assert compute_reward_v2(predicted, expected, observations=observations).dynamic_score == 0


@pytest.mark.parametrize("expression,credited", [
    ("1000/20", True), (" (1000 / 20) ", True),
    ("20/1000", False), ("1000+20", False), ("1000/5", False),
    ("1000", False), ("__dynamic__", False), ("1000/20 + 0", False),
])
def test_dynamic_density_expression_requires_both_observed_operands(expression, credited):
    expected = [call("lookup_db", table="countries", key="France"),
                call("calculator", expression="__dynamic__")]
    predicted = [expected[0], call("calculator", expression=expression)]
    observations = trace(predicted, [{"population": 1000, "area_km2": 20, "gdp_billion_usd": 5}, "50"])
    breakdown = compute_reward_v2(predicted, expected, observations=observations)
    assert (breakdown.dynamic_score > 0) is credited
    assert is_complete_tool_chain(predicted, expected, observations=observations) is credited


def test_three_step_dynamic_chain_copies_calculator_result():
    expected = [call("lookup_db", table="countries", key="France"),
                call("calculator", expression="__dynamic__"),
                call("convert_units", value="__dynamic__", from_unit="km", to_unit="miles")]
    predicted = [expected[0], call("calculator", expression="1000 / 20"),
                 call("convert_units", value=50.0, from_unit="KM", to_unit="Miles")]
    observations = trace(predicted, [{"population": 1000, "area_km2": 20}, "50.0", "31.0685"])
    assert compute_reward_v2(predicted, expected, observations=observations).total == pytest.approx(4)
    assert is_complete_tool_chain(predicted, expected, observations=observations)
    # Complete prescribed protocol is NOT a claim that the old density units
    # template is physically correct; that dataset limitation remains explicit.


@pytest.mark.parametrize("value", [15, 15.0, "15", " 15.0 "])
def test_static_numeric_and_unit_normalization(value):
    expected = [call("convert_units", value=15, from_unit="kg", to_unit="lbs")]
    predicted = [call("convert_units", value=value, from_unit="KG", to_unit="Lbs")]
    assert normalize_value(value) == normalize_value(15)
    assert compute_reward_v2(predicted, expected).total == pytest.approx(4)
    assert normalize_value(True) != normalize_value(1)


def test_expression_spacing_but_not_arbitrary_algebra_or_invalid_token_joining():
    expected = [call("calculator", expression="441+23")]
    spaced = [call("calculator", expression="441 + 23")]
    assert normalize_expression("441+23") == normalize_expression("441 + 23")
    assert compute_reward_v2(spaced, expected).total == pytest.approx(4)
    for expression in ("23+441", "44 1+23", "__import__('os').getcwd()"):
        assert compute_reward_v2([call("calculator", expression=expression)], expected).total <= 2


@pytest.mark.parametrize("result", [
    "Error: 'value' must be a number.", "Not found: country", {"error": "tool failed"},
    [], None, float("nan"), [{"text": "No results found for query", "score": 0}],
])
def test_explicit_tool_error_prevents_completion_and_above_threshold_reward(result):
    expected = [call("convert_units", value=15, from_unit="kg", to_unit="lbs")]
    observations = trace(expected, [result])
    assert not is_complete_tool_chain(expected, expected, observations=observations)
    breakdown = compute_reward_v2(expected, expected, observations=observations)
    assert breakdown.completion_score == 0
    assert breakdown.total <= 2


def test_execution_trace_is_required_by_frozen_checker_not_by_static_reward_unit_api():
    expected = [call("calculator", expression="3+5")]
    assert compute_reward_v2(expected, expected).total == pytest.approx(4)
    assert not is_complete_tool_chain(expected, expected)
    assert is_complete_tool_chain(expected, expected, observations=trace(expected, ["8"]))


def test_extra_calls_are_penalized_and_incomplete_reward_is_capped():
    expected = [call("get_weather", city="London") for _ in range(20)]
    predicted = expected + [expected[0]]
    breakdown = compute_reward_v2(predicted, expected)
    assert breakdown.completion_score == 0
    assert breakdown.cap_penalty < 0
    assert breakdown.total == 2


def test_breakdown_contributions_sum_to_total_and_dispatch_matches():
    expected = [call("get_weather", city="London"), call("get_weather", city="Mumbai")]
    for predicted in ([], expected[:1], expected, expected + expected):
        breakdown = compute_reward_v2(predicted, expected)
        values = asdict(breakdown)
        assert -3 <= breakdown.total <= 4
        assert sum(values[key] for key in (
            "baseline_score", "format_score", "tool_score", "sequence_score", "argument_key_score",
            "argument_value_score", "dynamic_score", "completion_score", "cap_penalty",
        )) == pytest.approx(breakdown.total)
        assert compute_reward_by_version(predicted, expected, version="v2") == breakdown.total


@pytest.mark.parametrize("bad", [{"name": [], "arguments": {}}, call("unknown_tool"),
                                  {"name": "calculator", "arguments": None}])
def test_malformed_calls_do_not_raise_or_receive_format_credit(bad):
    assert compute_reward_v2([bad], [call("calculator", expression="3+5")]).total == -3


def test_checker_rejects_extra_args_but_accepts_explicit_weather_default():
    expected = [call("get_weather", city="London")]
    default = [call("get_weather", city="london", units="Celsius")]
    assert is_complete_tool_chain(default, expected, observations=trace(default, [{"temperature": 12}]))
    for predicted in ([call("get_weather", city="London", units="fahrenheit")],
                      [call("get_weather", city="London", unused=True)]):
        assert not is_complete_tool_chain(predicted, expected, observations=trace(predicted, [{"temperature": 12}]))


def test_missing_dynamic_key_and_unsupported_route_receive_no_dynamic_credit():
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]
    predicted = [expected[0], call("convert_units", from_unit="celsius", to_unit="fahrenheit")]
    assert compute_reward_v2(predicted, expected, observations=trace(predicted, [{"temperature": 15}, "59"])).dynamic_score == 0
    assert score_final_answer("anything") == 0.0


def test_same_generation_cannot_claim_dynamic_observation_reuse():
    expected = [call("get_weather", city="London"),
                call("convert_units", value="__dynamic__", from_unit="celsius", to_unit="fahrenheit")]
    predicted = [expected[0], call("convert_units", value=15, from_unit="celsius", to_unit="fahrenheit")]
    observations = trace(predicted, [{"temperature": 15}, "59"])
    observations[0]["turn"] = observations[1]["turn"] = 1
    assert compute_reward_v2(predicted, expected, observations=observations).dynamic_score == 0
    assert not is_complete_tool_chain(predicted, expected, observations=observations)
    observations[1]["turn"] = 2
    assert compute_reward_v2(predicted, expected, observations=observations).total == pytest.approx(4)
    assert is_complete_tool_chain(predicted, expected, observations=observations)


def test_parallel_independent_calls_do_not_require_dynamic_dependencies():
    expected = [call("get_weather", city="London"), call("get_weather", city="Mumbai")]
    observations = trace(expected, [{"temperature": 12}, {"temperature": 32}])
    for entry in observations:
        entry["turn"] = 1
    assert is_complete_tool_chain(expected, expected, observations=observations)
    assert compute_reward_v2(expected, expected, observations=observations).total == pytest.approx(4)


def test_unknown_version_is_configuration_error():
    with pytest.raises(ValueError, match="Unknown reward version"):
        compute_reward_by_version([], [], version="v3")

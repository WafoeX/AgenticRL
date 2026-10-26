"""Release-only statistics checks; no GPU, tokenizer or raw feedback required."""

import csv

import pytest

from scripts.summarize_final_test import DEFAULT_CSV, summarize


def test_published_statistics():
    result = summarize(DEFAULT_CSV)
    assert list(result["answer_correct"].values()) == [144, 181, 171, 170]
    assert result["grpo_mean_percent"] == pytest.approx(72.5)
    assert result["grpo_sample_sd_pp"] == pytest.approx(2.53448439)
    assert result["mean_gain_pp"] == pytest.approx(12.5)
    assert result["paired_bootstrap_ci95_pp"] == pytest.approx([8.3333333333, 16.6666666667])
    assert result["mean_tool_calls_change_percent"] == pytest.approx(8.3743842365)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "nonbinary", "metadata", "calls"])
def test_reject_bad_cohort(tmp_path, mutation):
    with DEFAULT_CSV.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "unknown":
        rows[0]["arm"] = "unknown"
    elif mutation == "nonbinary":
        rows[0]["correct"] = "2"
    elif mutation == "metadata":
        rows[0]["family"] = "different-family"
    else:
        rows[0]["tool_calls"] = "-1"
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        summarize(path)

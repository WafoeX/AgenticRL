"""R3 contracts: synthetic train fixtures; no holdout labels/model inference."""

import copy
import os
import subprocess
from pathlib import Path

import pytest

from src.research_environment import canonical_json
from src.research_generator import PARTITIONS, make_instance, render_record
from src.research_schema import load_json
from src.research_sft import (
    build_corpus,
    build_expert,
    pad_batch,
    tokenize_expert,
    validate_expert,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_RECIPES = [r for r, split in PARTITIONS.items() if split == "train"]


def record(recipe="country_density_convert"):
    config = load_json(ROOT / "configs/research/dataset_v1.json")
    return render_record(config, make_instance(config, recipe, "base", 13),
                         {"split": "train", "generalization": "train", "wording": "train"})


@pytest.mark.parametrize("recipe", TRAIN_RECIPES)
def test_executed_expert(recipe):
    row = record(recipe)
    expert = build_expert(row)
    assert expert["checks"]["answer"]["correct"]
    assert expert["checks"]["trace"]["strict_complete"]
    assert len(expert["messages"]) == 3 + 2 * row["reference_depth"]
    assert [x["turn"] for x in expert["trace"]] == list(range(row["reference_depth"]))
    assert "__dynamic__" not in canonical_json(expert["messages"])


@pytest.mark.parametrize("split", ["validation", "test", None])
def test_reject_nontrain_before_processing(split):
    with pytest.raises(ValueError, match="train"):
        build_expert({"split": split})
    with pytest.raises(ValueError, match="train"):
        build_corpus([{"split": split}], None, max_length=4096)


@pytest.mark.parametrize("split", ["validation", "test", None])
def test_tokenization_rejects_holdout_before_tokenizer_load(split):
    with pytest.raises(ValueError, match="train"):
        tokenize_expert({"split": split}, None, max_length=4096)


@pytest.mark.parametrize("kind", ["observation", "arguments", "same_turn", "missing", "answer", "prompt", "id"])
def test_independent_rejects_corruption(kind):
    row = record()
    expert = build_expert(row)
    if kind == "observation":
        expert["trace"][0]["observation"] = "{}"
    elif kind == "arguments":
        expert["trace"][1]["arguments"] = {"expression": "1+1"}
    elif kind == "same_turn":
        expert["trace"][1]["turn"] = 0
    elif kind == "missing":
        expert["trace"].pop()
    elif kind == "answer":
        expert["messages"][-1]["content"] = '{"value":0,"unit":"people_per_mile2"}'
    elif kind == "prompt":
        expert["messages"][1]["content"] += " HIDDEN GOLD"
    else:
        expert["messages"][2]["tool_calls"][0]["id"] = "fake"
    with pytest.raises(ValueError):
        validate_expert(row, expert)


def test_saved_gold_cannot_override_independent_check():
    row = record()
    expert = build_expert(row)
    poisoned = copy.deepcopy(row)
    poisoned["oracle_answer"] = {"value": 0, "unit": "people_per_mile2"}
    poisoned["reference_trace"] = []
    poisoned["expected_calls"] = []
    assert validate_expert(poisoned, expert)["answer"]["correct"]
    expert["messages"][-1]["content"] = canonical_json(poisoned["oracle_answer"])
    with pytest.raises(ValueError):
        validate_expert(poisoned, expert)


def test_duplicate_groups_rejected():
    row = record()
    with pytest.raises(ValueError, match="duplicate"):
        build_corpus([row, copy.deepcopy(row)], None, max_length=4096)


def test_padding_mask():
    rows = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]}]
    batch = pad_batch(rows, pad_token_id=0)
    assert batch["labels"] == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"][1] == [1, 1, 0]
    assert batch["input_ids"][1] == [4, 5, 0]
    with pytest.raises(ValueError):
        pad_batch([], pad_token_id=0)


def test_real_local_qwen_tokenizer_contract():
    """Tokenizer only in existing isolated environment; explicitly skip if absent."""
    python = Path(os.environ.get("RESEARCH_PILOT_PYTHON", str(Path.home() / ".cache/agenticrl-pilot-venv/bin/python")))
    base = Path(os.environ.get("RESEARCH_TOKENIZER_DIR", str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")))
    if not python.exists() or not (base / "tokenizer_config.json").exists():
        if "RESEARCH_PILOT_PYTHON" in os.environ or "RESEARCH_TOKENIZER_DIR" in os.environ:
            pytest.fail("explicit contract-test Python/tokenizer path unavailable")
        pytest.skip("local pinned tokenizer/pilot environment not available")
    code = '''
import copy
from transformers import AutoTokenizer
from src.research_environment import TOOL_SCHEMAS
from src.research_generator import make_instance, render_record
from src.research_schema import load_json
from src.research_sft import build_expert, tokenize_expert, build_corpus, pad_batch
import sys
config = load_json(__import__('pathlib').Path('configs/research/dataset_v1.json'))
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
rows = []
for recipe in ('simple_math', 'age_calculation', 'country_density_convert', 'weather_compare'):
    record = render_record(config, make_instance(config, recipe, 'base', 13),
                           {'split':'train','generalization':'train','wording':'train'})
    expert = build_expert(record)
    row = tokenize_expert(expert, tok, max_length=4096)
    assert len(row['assistant_token_ranges']) == record['reference_depth'] + 1
    n = row['token_count']
    assert tokenize_expert(expert, tok, max_length=n) == row
    for cap in (1, n-1):
        try: tokenize_expert(expert, tok, max_length=cap)
        except ValueError as e: assert 'overlength' in str(e)
        else: raise AssertionError('silent truncation')
    text = tok.apply_chat_template(expert['messages'], tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=False)
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    # Every observation token is ignored, including the tool-response wrapper.
    for message in expert['messages']:
        if message['role'] == 'tool':
            needle = '<tool_response>\\n' + message['content'] + '\\n</tool_response>'
            start = text.index(needle)
            end = start + len(needle)
            assert all(label == -100 for label, (a,b) in zip(row['labels'], offsets) if a < end and b > start)
    for a,b in row['assistant_token_ranges']:
        assert row['input_ids'][b-1] == tok.eos_token_id
        assert all(row['labels'][j] == row['input_ids'][j] for j in range(a,b))
    supervised = {j for a,b in row['assistant_token_ranges'] for j in range(a,b)}
    assert all(label == -100 for j,label in enumerate(row['labels']) if j not in supervised)
    assert row['labels'][0] == -100 and row['supervised_tokens'] > 0
    experts, tokens, summary = build_corpus([record], tok, max_length=n-1)
    assert not experts and not tokens and summary['rejected_count'] == 1
    rows.append(row)
batch = pad_batch(rows, pad_token_id=tok.pad_token_id)
assert all(label == -100 for labels,mask in zip(batch['labels'],batch['attention_mask']) for label,m in zip(labels,mask) if m == 0)
print('real tokenizer: depths 1/2/3, EOS, context masks, exact length, whole-example rejection and padding passed')
'''
    result = subprocess.run([str(python), "-c", code, str(base)], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout

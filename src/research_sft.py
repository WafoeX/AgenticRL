"""Train-only executed experts and fail-closed assistant-only SFT preprocessing.

No model loading, optimization, test access, packing or lossy truncation.
"""

import copy
from collections import Counter

from .research_checker import check_messages
from .research_environment import TOOL_SCHEMAS, canonical_json
from .research_generator import execute_reference
from .research_schema import model_messages, validate_record

VERSION = "research-sft-data-v1"
IGNORE_INDEX = -100


def validate_expert(record: dict, expert: dict) -> dict:
    """Rebuild public chat from the trace; independently replay/answer-check it."""
    if record["split"] != "train" or expert.get("split") != "train":
        raise ValueError("SFT requires train only")
    if expert.get("sample_id") != record["sample_id"]:
        raise ValueError("expert source identity mismatch")
    messages = model_messages(record)
    for i, step in enumerate(expert["trace"]):
        if step["turn"] != i:
            raise ValueError("each executed call requires its own assistant turn")
        call_id = f"expert-turn-{i}-call-0"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": step["name"], "arguments": step["arguments"]}}]},
            {"role": "tool", "tool_call_id": call_id, "content": step["observation"]},
        ])
    final = expert["messages"][-1]
    if set(final) != {"role", "content"} or final["role"] != "assistant":
        raise ValueError("invalid terminal answer")
    messages.append(final)
    if messages != expert["messages"]:
        raise ValueError("chat/trace mismatch or hidden prompt metadata")
    checked = check_messages(record, messages, expert["trace"])
    trace = checked["trace"]
    if (not checked["answer"]["correct"] or not trace["execution_valid"]
            or not trace["strict_complete"] or not trace["semantic_valid"]
            or trace["dynamic_status"] not in {"valid", "not_applicable"}):
        raise ValueError("independent expert check failed: " + canonical_json(checked))
    return checked


def build_expert(record: dict) -> dict:
    if record.get("split") != "train":
        raise ValueError("SFT requires train only")
    validate_record(record, replay=False)
    # Saved reference/gold/expected_calls are never used to construct supervision.
    trace, _, answer = execute_reference({"recipe": record["recipe_id"],
                                         "parameters": record["parameters"],
                                         "world": record["environment_snapshot"]})
    messages = model_messages(record)
    for i, step in enumerate(trace):
        call_id = f"expert-turn-{i}-call-0"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": step["name"], "arguments": step["arguments"]}}]},
            {"role": "tool", "tool_call_id": call_id, "content": step["observation"]},
        ])
    messages.append({"role": "assistant", "content": canonical_json(answer)})
    expert = {"version": VERSION, "split": "train", "sample_id": record["sample_id"],
              "semantic_group_id": record["semantic_group_id"], "recipe_id": record["recipe_id"],
              "reference_depth": record["reference_depth"], "messages": messages, "trace": trace}
    expert["checks"] = validate_expert(record, expert)
    return expert


def tokenize_expert(expert: dict, tokenizer, *, max_length: int) -> dict:
    """Exact local template. Mask headers/context, supervise assistant payload + EOS.

    Compute character spans from prefix renders, then align against ONE full tokenization.
    Reject boundary-straddling tokens, non-prefix templates, lost EOS and overlength
    examples rather than silently truncating a dependency or terminal answer.
    """
    if expert.get("split") != "train":
        raise ValueError("SFT requires train only")
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("invalid max_length")
    if not tokenizer.is_fast or not tokenizer.chat_template or tokenizer.eos_token != "<|im_end|>":
        raise ValueError("exact fast Qwen tokenizer with im_end EOS required")
    messages = expert["messages"]
    if messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("terminal assistant answer required")

    def render(chat, generation=False):
        return tokenizer.apply_chat_template(chat, tools=copy.deepcopy(TOOL_SCHEMAS),
                                             tokenize=False, add_generation_prompt=generation)

    text = render(messages)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix, through = render(messages[:i], True), render(messages[:i + 1])
        if not text.startswith(prefix) or not text.startswith(through) or not through.endswith(tokenizer.eos_token + "\n"):
            raise ValueError("template is not prefix stable or lacks assistant EOS")
        start, end = len(prefix), len(through) - 1
        if end <= start:
            raise ValueError("empty assistant target")
        spans.append((start, end))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    # Assert equivalence to the evaluation template's native tokenization.
    if ids != tokenizer.apply_chat_template(messages, tools=copy.deepcopy(TOOL_SCHEMAS), tokenize=True, add_generation_prompt=False):
        raise ValueError("template tokenization differs")
    if len(ids) > max_length:
        raise ValueError(f"overlength: {len(ids)} > {max_length}; whole example rejected, no truncation")
    labels, ranges = [IGNORE_INDEX] * len(ids), []
    for start, end in spans:
        selected = []
        for j, (left, right) in enumerate(offsets):
            if left < end and right > start:
                if left < start or right > end or right <= left:
                    raise ValueError("token straddles supervision boundary")
                labels[j] = ids[j]
                selected.append(j)
        if (not selected or selected[0] == 0 or ids[selected[-1]] != tokenizer.eos_token_id
                or offsets[selected[0]][0] != start or offsets[selected[-1]][1] != end):
            raise ValueError("assistant span/EOS alignment failed")
        ranges.append([selected[0], selected[-1] + 1])
    if len(ranges) != expert["reference_depth"] + 1:
        raise ValueError("lost tool consumption/final turn")
    return {"sample_id": expert["sample_id"], "input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": labels, "assistant_token_ranges": ranges,
            "token_count": len(ids), "supervised_tokens": sum(x != IGNORE_INDEX for x in labels)}


def pad_batch(rows: list[dict], *, pad_token_id: int) -> dict:
    """Right padding, no packing; labels retain causal-LM unshifted convention."""
    if not rows:
        raise ValueError("empty batch")
    width = max(len(row["input_ids"]) for row in rows)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for row in rows:
        size = len(row["input_ids"])
        if len(row["labels"]) != size or row["attention_mask"] != [1] * size:
            raise ValueError("invalid unpadded row")
        for key, fill in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", IGNORE_INDEX)):
            result[key].append(row[key] + [fill] * (width - size))
    return result


def build_corpus(records: list[dict], tokenizer, *, max_length: int) -> tuple[list, list, dict]:
    if not records or any(r.get("split") != "train" for r in records):
        raise ValueError("nonempty train-only corpus required")
    if len({r["sample_id"] for r in records}) != len(records) or len({r["semantic_group_id"] for r in records}) != len(records):
        raise ValueError("duplicate source sample/group")
    experts, tokens, rejected = [], [], []
    for record in records:
        try:
            expert = build_expert(record)
            tokenized = tokenize_expert(expert, tokenizer, max_length=max_length)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"sample_id": record["sample_id"], "reason": str(exc)})
            continue
        experts.append(expert)
        tokens.append(tokenized)
    lengths = sorted(t["token_count"] for t in tokens)
    summary = {"source_count": len(records), "accepted": len(experts), "rejected_count": len(rejected),
               "rejected": rejected, "depth_counts": dict(Counter(e["reference_depth"] for e in experts)),
               "recipe_counts": dict(Counter(e["recipe_id"] for e in experts)),
               "token_length": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                                "total": sum(lengths)},
               "supervised_tokens": sum(t["supervised_tokens"] for t in tokens)}
    return experts, tokens, summary

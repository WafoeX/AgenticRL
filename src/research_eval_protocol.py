"""Versioned development evaluation contract and complete attempt schedule."""

import copy
import hashlib
from pathlib import Path

from .research_answers import reject_constant, unique_object
from .research_environment import TOOL_SCHEMAS, canonical_json, numeric
from .research_model_identity import file_hash
from .research_schema import SYSTEM_PROMPT, digest, load_json, validate_record

PROTOCOL_VERSION = "research-eval-v1"


def validate_protocol(protocol: dict) -> None:
    fields = {"protocol_version", "modes", "sample_k", "seed", "deterministic", "sample", "max_turns", "max_calls_per_turn",
              "max_new_tokens_per_turn", "max_output_tokens_per_attempt", "max_prompt_tokens", "request_timeout_seconds",
              "concurrency", "chat_template_policy", "tool_codec", "infra_policy", "test_policy"}
    if not isinstance(protocol, dict) or set(protocol) != fields or protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("invalid evaluation protocol fields/version")
    modes = protocol["modes"]
    if modes not in [["deterministic"], ["sample"], ["deterministic", "sample"]]:
        raise ValueError("modes must be deterministic/sample/both in fixed order")
    limits = {"sample_k": (8, 8), "seed": (0, 2**32 - 1), "max_turns": (1, 6), "max_calls_per_turn": (1, 8),
              "max_new_tokens_per_turn": (1, 8192), "max_output_tokens_per_attempt": (1, 49152),
              "max_prompt_tokens": (1, 32768), "concurrency": (1, 16)}
    for key, (minimum, maximum) in limits.items():
        if type(protocol[key]) is not int or not minimum <= protocol[key] <= maximum:
            raise ValueError(f"invalid protocol limit: {key}")
    if not 0 < numeric(protocol["request_timeout_seconds"]) <= 3600:
        raise ValueError("invalid request timeout")
    for mode in ("deterministic", "sample"):
        params = protocol[mode]
        if not isinstance(params, dict) or set(params) != {"temperature", "top_p"}:
            raise ValueError("invalid decoding parameters")
        temperature, top_p = numeric(params["temperature"]), numeric(params["top_p"])
        if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
            raise ValueError("decoding parameters out of range")
    if protocol["deterministic"] != {"temperature": 0.0, "top_p": 1.0} or protocol["sample"]["temperature"] <= 0:
        raise ValueError("deterministic must be greedy, sample must have positive temperature")
    constants = {"chat_template_policy": "exact-local-tokenizer-no-override", "tool_codec": "qwen-tool-tags-v1",
                 "infra_policy": "retain-all-no-retry-invalidate-comparison", "test_policy": "development-only-reject-test"}
    if any(protocol[key] != value for key, value in constants.items()):
        raise ValueError("unsupported template/codec/failure/test policy")
    canonical_json(protocol)


def protocol_identity(protocol: dict) -> dict:
    validate_protocol(protocol)
    return {"version": PROTOCOL_VERSION, "configuration": copy.deepcopy(protocol), "sha256": digest(protocol),
            "system_prompt_sha256": digest(SYSTEM_PROMPT), "tool_schema_sha256": digest(TOOL_SCHEMAS),
            "seed_derivation": "sha256(protocol_seed,sample_id,mode,index,turn) first 8 hex",
            "determinism_note": "temperature 0 is greedy, not cross-environment byte determinism"}


def load_development_records(directory: Path, *, split: str = "validation", limit: int | None = None) -> tuple[list[dict], dict]:
    import json

    if split not in {"train", "validation"}:
        raise ValueError("research evaluation is development-only; test forbidden")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("development limit must be a positive integer")
    directory = directory.resolve(strict=True)
    manifest = load_json(directory / "manifest.json")
    path = directory / f"{split}.jsonl"
    expected = manifest["files"][path.name]
    if file_hash(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
        raise ValueError("dataset bytes differ from manifest")
    try:
        rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError("invalid dataset JSONL") from exc
    if not rows or len(rows) != manifest["counts"][split]:
        raise ValueError("dataset count differs from manifest")
    for row in rows:
        validate_record(row)
        if row["split"] != split:
            raise ValueError("mixed/incorrect input split")
    if len({row["sample_id"] for row in rows}) != len(rows) or len({row["semantic_group_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source scenario/group")
    if limit is not None and limit > len(rows):
        raise ValueError("development limit exceeds dataset; refusing implicit clamping")
    selected = rows if limit is None else rows[:limit]
    provenance = {"path": str(path), "sha256": expected["sha256"], "manifest_sha256": file_hash(directory / "manifest.json"),
                  "dataset_version": manifest["dataset_version"], "split": split, "source_count": len(rows),
                  "selected_count": len(selected), "selection": "all" if limit is None else f"explicit-first-{limit}-development-subset",
                  "selected_sample_ids": [r["sample_id"] for r in selected]}
    return selected, provenance


def plan_attempts(records: list[dict], protocol: dict) -> list[dict]:
    validate_protocol(protocol)
    if not records or any(r["split"] not in {"train", "validation"} for r in records):
        raise ValueError("nonempty development records required; test forbidden")
    if len({r["sample_id"] for r in records}) != len(records) or len({r["semantic_group_id"] for r in records}) != len(records):
        raise ValueError("duplicate scenario/group")
    result = []
    for scenario_index, row in enumerate(records):
        for mode in protocol["modes"]:
            for index in range(1 if mode == "deterministic" else protocol["sample_k"]):
                slot = {"sample_id": row["sample_id"], "semantic_group_id": row["semantic_group_id"], "scenario_index": scenario_index,
                        "mode": mode, "index": index, "family": row["family"], "reference_depth": row["reference_depth"],
                        "dynamic_dependency": row["dynamic_dependency"], "generalization": row["generalization"], "split": row["split"]}
                slot["attempt_id"] = "attempt-" + digest([row["sample_id"], mode, index, digest(protocol)])
                result.append(slot)
    return result


def request_seed(protocol: dict, slot: dict, turn: int) -> int:
    payload = canonical_json([protocol["seed"], slot["sample_id"], slot["mode"], slot["index"], turn])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)

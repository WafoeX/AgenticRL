"""Operational holdout sealing; stdlib only. Never parse test labels or run a model."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .research_model_identity import file_hash
from .research_schema import load_json

POLICY = "configs/research/preregistration_v1.json"
LOCK = "configs/research/preregistration_v1.lock.json"
COUNTS = {"train": 1200, "validation": 180, "test": 240}
DATA_FILES = {"manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl", "audit.json", "DATA_CARD.md"}
# Approved pre-freeze candidate identity, not merely hashes supplied by a mutable manifest.
APPROVED_DATA_SHA256 = {
    "DATA_CARD.md": "7464ca7db9e9aa30ba3cec0ad2797e0ec5f9030ea39291d942107549b8a089b8",
    "audit.json": "997301b7fcf626a469451d81e8b3afd2c8d10f467e8f62b6ee33021a8e02acd8",
    "manifest.json": "c8901d81e25fd80c600525c4f53ac7dcec2e3cf21bee0cb340c8d9818c3a4dde",
    "test.jsonl": "08040a4e2e7b7c410f17937496a22bccd09f925e4c5dba35da2f027380410a52",
    "train.jsonl": "b98fc61cfae88426e065ff817db9757650944c76f3268298de82eb76a097048e",
    "validation.jsonl": "941132e1c93ce26cba6417eac7892c11b7c64c2fc14d3c21b02902e1dff70261",
}
SOURCE_PATHS = [
    "configs/research/dataset_v1.json", "configs/research/schema_v1.json",
    "configs/research/evaluation_v1.json", "scripts/generate_research_dataset.py",
    "src/research_schema.py", "src/research_generator.py", "src/research_audit.py",
    "src/research_environment.py", "src/research_answers.py", "src/research_checker.py",
    "src/research_model_identity.py", "src/research_eval_protocol.py", "src/research_eval_backend.py",
    "src/research_evaluation.py", "src/research_eval_metrics.py", "scripts/eval_research.py",
    "src/research_registration.py", "scripts/freeze_research.py", "tests/test_research_registration.py",
    "docs/research_preregistration_v1_zh.md",
]


def local_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe registration path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("registration path escapes root or is a symlink")
    return path


def inventory(directory: Path, names: set[str]) -> dict:
    result = {}
    for name in sorted(names):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing/symlink frozen artifact: " + name)
        result[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return result


def explicit_false(value: object) -> bool:
    return type(value) is bool and not value


def policy_at(root: Path) -> dict:
    policy = load_json(local_path(root, POLICY))
    if (policy.get("registration_id") != "research-preregistration-v1-20261012"
            or policy.get("status") != "decisions-locked-training-budget-annex-required"
            or policy.get("candidate_directory") != "data/research_v1_candidate"
            or policy.get("frozen_directory") != "data/research_v1_frozen"
            or type(policy.get("seed")) is not int or policy["seed"] != 20261011
            or policy.get("seed_count") != 1
            or policy.get("arms") != ["B0", "B1", "G1", "G2", "A1"]):
        raise ValueError("unapproved registration identity/seed/arms/paths")
    budget = policy.get("budget", {})
    test = policy.get("final_test", {})
    if (not explicit_false(budget.get("formal_training_authorized"))
            or not explicit_false(budget.get("GPU_authorized_this_registration"))
            or budget.get("total_GPU_hours_cap") != 20 or not budget.get("pending_annex")
            or not explicit_false(test.get("development_inference_allowed"))
            or test.get("planned_attempts_single_seed") != 240 * 9 * 5):
        raise ValueError("registration must remain closed to training/GPU/development test")
    return policy


def verify_data(directory: Path) -> dict:
    manifest = load_json(directory / "manifest.json")
    if manifest.get("counts") != COUNTS or manifest.get("test_status") != "candidate_not_frozen":
        raise ValueError("unexpected original dataset identity/counts")
    actual = inventory(directory, DATA_FILES)
    if {name: info["sha256"] for name, info in actual.items()} != APPROVED_DATA_SHA256:
        raise ValueError("dataset differs from approved pre-freeze candidate identity")
    if set(manifest["files"]) != DATA_FILES - {"manifest.json"}:
        raise ValueError("unexpected original artifact inventory")
    if any(actual[name] != info for name, info in manifest["files"].items()):
        raise ValueError("dataset artifact bytes differ from original manifest")
    for split, count in COUNTS.items():
        # Count raw lines only; no test task, facts, gold or trace are inspected.
        with (directory / f"{split}.jsonl").open("rb") as stream:
            if sum(1 for _ in stream) != count:
                raise ValueError("dataset line count mismatch")
    return actual


def source_inventory(root: Path) -> dict:
    return {name: file_hash(local_path(root, name)) for name in SOURCE_PATHS}


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def create_registration(root: Path) -> dict:
    root = root.resolve(strict=True)
    policy = policy_at(root)
    candidate = local_path(root, policy["candidate_directory"])
    frozen = local_path(root, policy["frozen_directory"])
    lock_path = local_path(root, LOCK)
    if frozen.exists() or lock_path.exists():
        raise ValueError("registration exists; refusing overwrite/resume/reseal")
    if {p.name for p in candidate.iterdir()} != DATA_FILES:
        raise ValueError("candidate directory contains missing/extra artifacts")
    before = verify_data(candidate)
    source_before = source_inventory(root)
    policy_sha = file_hash(root / POLICY)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".research-freeze-", dir=frozen.parent))
    try:
        for name in sorted(DATA_FILES):
            shutil.copyfile(candidate / name, staging / name)
        if verify_data(staging) != before or verify_data(candidate) != before:
            raise ValueError("candidate changed while freezing")
        freeze = {"version": "research-operational-freeze-v1", "registration_id": policy["registration_id"],
                  "release_id": "research-v1-frozen-20261012", "original_record_version": "research-v1-candidate",
                  "counts": COUNTS, "files": before, "preregistration_sha256": policy_sha,
                  "test_access": "closed-no-model-inference-until-separate-final-cohort-authorization",
                  "history": policy["test_history"], "source_hashes": source_before,
                  "note": "Original six files preserved byte-for-byte, including candidate manifest; this separate freeze record is release authority, not cryptographic access control."}
        write_new_json(staging / "freeze.json", freeze)
        if source_inventory(root) != source_before or file_hash(root / POLICY) != policy_sha:
            raise ValueError("registration source/policy changed while freezing")
        # No automatic regeneration/check can overwrite a sealed release.
        staging.rename(frozen)
        lock = {"version": "research-registration-lock-v1", "registration_id": policy["registration_id"],
                "policy_sha256": policy_sha, "freeze_sha256": file_hash(frozen / "freeze.json"),
                "frozen_directory": policy["frozen_directory"], "source_hashes": source_before,
                "formal_training_ready": False, "final_test_ready": False}
        write_new_json(lock_path, lock)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return verify_registration(root)


def verify_registration(root: Path) -> dict:
    root = root.resolve(strict=True)
    lock = load_json(local_path(root, LOCK))
    if file_hash(local_path(root, POLICY)) != lock["policy_sha256"]:
        raise ValueError("locked preregistration bytes changed")
    policy = policy_at(root)
    if (lock["version"] != "research-registration-lock-v1"
            or lock["registration_id"] != policy["registration_id"]
            or lock["frozen_directory"] != policy["frozen_directory"]
            or not explicit_false(lock["formal_training_ready"]) or not explicit_false(lock["final_test_ready"])):
        raise ValueError("registration lock identity/gates mismatch")
    frozen = local_path(root, policy["frozen_directory"])
    if {p.name for p in frozen.iterdir()} != DATA_FILES | {"freeze.json"}:
        raise ValueError("missing/extra frozen artifacts")
    freeze_path = local_path(root, policy["frozen_directory"] + "/freeze.json")
    if file_hash(freeze_path) != lock["freeze_sha256"]:
        raise ValueError("locked freeze metadata changed")
    freeze = load_json(freeze_path)
    if (freeze["registration_id"] != policy["registration_id"]
            or freeze["preregistration_sha256"] != lock["policy_sha256"]
            or freeze["files"] != verify_data(frozen)
            or freeze["source_hashes"] != lock["source_hashes"]
            or source_inventory(root) != lock["source_hashes"]):
        raise ValueError("frozen bytes/source identity mismatch; versioned amendment required")
    return {"registration_id": policy["registration_id"], "counts": COUNTS,
            "frozen_directory": policy["frozen_directory"], "test_sha256": freeze["files"]["test.jsonl"]["sha256"],
            "decision_registration_locked": True, "test_operationally_frozen": True,
            "formal_training_ready": False, "final_test_ready": False,
            "test_labels_parsed": False, "GPU_started": False, "model_inference": False}


def load_registered_development(root: Path, *, split: str = "validation", limit: int | None = None) -> tuple[list[dict], dict]:
    if split not in {"train", "validation"}:
        raise ValueError("registered development access rejects test")
    receipt = verify_registration(root)
    from .research_eval_protocol import load_development_records
    return load_development_records(root / receipt["frozen_directory"], split=split, limit=limit)

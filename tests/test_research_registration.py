"""Seal/tamper/closed-test contracts; no model inference, no test label inspection."""

import json
import shutil
from pathlib import Path

import pytest

from src.research_eval_protocol import load_development_records, plan_attempts
from src.research_registration import (
    DATA_FILES,
    LOCK,
    POLICY,
    SOURCE_PATHS,
    create_registration,
    load_registered_development,
    local_path,
    verify_registration,
)
from src.research_schema import load_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    for name in [POLICY, *SOURCE_PATHS]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    shutil.copytree(ROOT / "data/research_v1_candidate", tmp_path / "data/research_v1_candidate")
    return tmp_path


def test_byte_preserving_seal_and_registered_validation(workspace, monkeypatch):
    original_read_text = Path.read_text

    def no_test_label_text(path, *args, **kwargs):
        if path.name == "test.jsonl":
            raise AssertionError("freezing must not parse test labels")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_test_label_text)
    receipt = create_registration(workspace)
    assert receipt["test_operationally_frozen"] and not receipt["formal_training_ready"]
    assert not receipt["final_test_ready"] and not receipt["model_inference"]
    assert receipt["counts"] == {"train": 1200, "validation": 180, "test": 240}
    frozen = workspace / receipt["frozen_directory"]
    candidate = workspace / "data/research_v1_candidate"
    for name in DATA_FILES:
        assert (frozen / name).read_bytes() == (candidate / name).read_bytes()
    rows, provenance = load_registered_development(workspace, split="validation", limit=1)
    assert len(rows) == 1 and provenance["split"] == "validation"
    assert "research_v1_frozen" in provenance["path"]
    assert verify_registration(workspace) == receipt


@pytest.mark.parametrize("target", ["test.jsonl", "train.jsonl", "validation.jsonl", "manifest.json", "freeze.json", "extra"])
def test_frozen_artifact_tampering_is_rejected(workspace, target):
    create_registration(workspace)
    frozen = workspace / "data/research_v1_frozen"
    path = frozen / target
    path.write_bytes(path.read_bytes() + b"\n" if path.exists() else b"extra")
    with pytest.raises(ValueError):
        verify_registration(workspace)


@pytest.mark.parametrize("target", [POLICY, "src/research_checker.py", "configs/research/evaluation_v1.json"])
def test_policy_and_locked_source_changes_require_amendment(workspace, target):
    create_registration(workspace)
    path = workspace / target
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        verify_registration(workspace)


@pytest.mark.parametrize("target", ["test.jsonl", "freeze.json"])
def test_missing_frozen_artifact_is_rejected(workspace, target):
    create_registration(workspace)
    (workspace / "data/research_v1_frozen" / target).unlink()
    with pytest.raises(ValueError):
        verify_registration(workspace)


def test_refuses_overwrite_and_keeps_original_bytes(workspace):
    create_registration(workspace)
    original = (workspace / LOCK).read_bytes()
    with pytest.raises(ValueError, match="refusing overwrite"):
        create_registration(workspace)
    assert (workspace / LOCK).read_bytes() == original


def test_corrupt_candidate_does_not_create_frozen_release(workspace):
    path = workspace / "data/research_v1_candidate/test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        create_registration(workspace)
    assert not (workspace / LOCK).exists()
    assert not (workspace / "data/research_v1_frozen").exists()


@pytest.mark.parametrize("split", ["test", "unknown"])
def test_registered_development_rejects_test_before_reading_any_files(tmp_path, split):
    with pytest.raises(ValueError, match="rejects test"):
        load_registered_development(tmp_path, split=split)


def test_existing_entry_still_cannot_load_or_plan_test(workspace):
    create_registration(workspace)
    with pytest.raises(ValueError, match="test forbidden"):
        load_development_records(workspace / "data/research_v1_frozen", split="test")
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    with pytest.raises(ValueError, match="test forbidden"):
        plan_attempts([{"split": "test"}], protocol)


@pytest.mark.parametrize("location", ["budget.formal_training_authorized", "budget.GPU_authorized_this_registration", "final_test.development_inference_allowed"])
def test_registration_cannot_authorize_training_GPU_or_development_test(workspace, location):
    policy = load_json(workspace / POLICY)
    parent, key = location.split(".")
    policy[parent][key] = True
    (workspace / POLICY).write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="closed"):
        create_registration(workspace)


@pytest.mark.parametrize("path", ["../outside", "/outside-registration-root"])
def test_paths_cannot_escape_workspace(tmp_path, path):
    with pytest.raises(ValueError, match="unsafe"):
        local_path(tmp_path, path)


def test_candidate_and_frozen_symlinks_are_rejected(workspace):
    candidate = workspace / "data/research_v1_candidate"
    original = (candidate / "test.jsonl").read_bytes()
    target = workspace / "replacement-test"
    target.write_bytes(original)
    (candidate / "test.jsonl").unlink()
    (candidate / "test.jsonl").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        create_registration(workspace)


@pytest.mark.parametrize("target", ["freeze.json", "test.jsonl"])
def test_frozen_symlink_even_with_identical_bytes_is_rejected(workspace, target):
    create_registration(workspace)
    path = workspace / "data/research_v1_frozen" / target
    replacement = workspace / "replacement"
    replacement.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(replacement)
    with pytest.raises(ValueError, match="symlink"):
        verify_registration(workspace)


def test_coordinated_candidate_manifest_rewrite_is_not_approved(workspace):
    path = workspace / "data/research_v1_candidate/manifest.json"
    manifest = load_json(path)
    manifest["reproducibility"] = "coordinated modification before seal"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="approved pre-freeze"):
        create_registration(workspace)

"""Build one immutable CPU-prepared final-evaluation package; never open test labels."""
import argparse
import json
import zipfile
from pathlib import Path

from src.research_model_identity import file_hash
from src.research_registration import LOCK as REG_LOCK
from src.research_registration import POLICY, SOURCE_PATHS, verify_registration
from src.research_schema import load_json

ROOT = Path(__file__).resolve().parents[1]
LOCK = "configs/research/final_test_v1.lock.json"
NAMES = [
    "scripts/__init__.py", "scripts/eval_research_baselines.py", "scripts/train_research_sft.py",
    "scripts/pilot_research_sft.py", "scripts/eval_research_final_test_v1.py",
    "scripts/build_research_final_test_bundle_v1.py", "scripts/verify_research_eval_cpu.py",
    "scripts/diagnose_research.py",
    "configs/research/final_test_v1.json", "tests/test_research_final_test_v1.py",
    "tests/test_research_final_test_campaign_v1.py",
    "requirements-sft-pilot.txt", "docs/research_final_test_plan_v1_zh.md",
    "docs/colab_research_final_test_v1_zh.md",
]


def build(output):
    verify_registration(ROOT)
    names = sorted(set(NAMES + SOURCE_PATHS + [POLICY, REG_LOCK, "src/__init__.py"]
                       + [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").glob("research*.py")]
                       + [p.relative_to(ROOT).as_posix() for p in (ROOT / "data/research_v1_frozen").iterdir()]))
    hashes = {n: file_hash(ROOT / n) for n in names}
    lock_path = ROOT / LOCK
    lock = {"version": "four-model-final-test-source-lock-v1", "files": hashes,
            "test_labels_parsed": False, "GPU_executed": False, "preparation_revision": 2,
            "supersedes_failed_preparation_lock_sha256": "6a4bf361a6d1285715dce6b6bda2e6c21a4a36adb3e3dcf502e1a7a4580d02e6"}
    if lock_path.exists():
        if load_json(lock_path) != lock:
            raise ValueError("frozen package changed; do not silently reseal")
    else:
        with lock_path.open("x") as stream:
            json.dump(lock, stream, sort_keys=True, indent=2)
            stream.write("\n")
    files = {n: (ROOT / n).read_bytes() for n in names + [LOCK]}
    # Historical metadata only: no pretrained weights or test labels are parsed.
    feedback = ROOT / "results/sft-to30-feedback.zip"
    try:
        with zipfile.ZipFile(feedback) as z:
            best = json.loads(z.read("run/selection.json"))["best"]["identity"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("cannot read historical SFT identity") from exc
    restore = {"revision": best["origin"]["revision"], "base_files": best["base_files"],
               "base_fingerprint": best["base_fingerprint"], "source_feedback_sha256": file_hash(feedback)}
    files["final_test_base_restore.json"] = (json.dumps(restore, sort_keys=True, indent=2) + "\n").encode()
    import hashlib
    manifest = {"version": "four-model-final-test-v1", "GPU_execution_authorized": False,
                "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}}
    files["bundle_manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    sha = file_hash(output)
    output.with_suffix(output.suffix + ".sha256").write_text(sha + "  " + output.name + "\n")
    print(json.dumps({"path": str(output), "files": len(files), "bytes": output.stat().st_size, "sha256": sha}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output.resolve())

"""Local safetensors identity contracts. Hash bytes, never select 'latest' or download."""

import hashlib
import re
from pathlib import Path

from .research_schema import digest, load_json

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
IDENTITY_VERSION = "research-model-identity-v1"


def file_hash(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash model artifact: {path.name}") from exc


def artifact_inventory(directory: Path) -> dict:
    """Flat HF artifacts only. HF snapshot->same repo blobs symlinks are allowed."""
    if not directory.is_dir():
        raise ValueError("explicit model directory does not exist")
    hashes = {}
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue
        relevant = path.suffix in {".json", ".safetensors", ".jinja", ".txt", ".model"}
        if path.suffix in {".bin", ".pt", ".pth", ".py"}:
            raise ValueError("pickle weights/custom code forbidden; use local safetensors")
        if not relevant:
            continue
        target = path.resolve(strict=True)
        if path.is_symlink() and not target.is_relative_to(directory):
            cache_blobs = directory.parent.parent / "blobs"
            if directory.parent.name != "snapshots" or not target.is_relative_to(cache_blobs):
                raise ValueError("model symlink escapes directory/HF repo blob cache")
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError("empty/non-file model artifact")
        hashes[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if not hashes:
        raise ValueError("empty model directory")
    return hashes


def verify_weights(directory: Path, files: dict, stem: str) -> None:
    single, index = stem + ".safetensors", stem + ".safetensors.index.json"
    if single in files and index in files:
        raise ValueError("ambiguous single/sharded weights")
    if single not in files and index not in files:
        raise ValueError("missing safetensors weights")
    expected = {single}
    if index in files:
        mapping = load_json(directory / index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("invalid safetensors shard index")
        expected = set(mapping.values())
        if any(Path(name).name != name or not name.startswith(stem + "-") or not name.endswith(".safetensors") or name not in files for name in expected):
            raise ValueError("missing/unsafe shard")
    actual = {name for name in files if name.endswith(".safetensors")}
    if actual != expected:
        raise ValueError("extra/missing/ambiguous tensor files")


def local_chat_template(directory: Path) -> str:
    if (directory / "chat_template.jinja").is_file():
        template = (directory / "chat_template.jinja").read_text(encoding="utf-8")
    else:
        template = load_json(directory / "tokenizer_config.json").get("chat_template")
        if isinstance(template, list):
            template = {item["name"]: item["template"] for item in template}
        if isinstance(template, dict):
            template = template.get("tool_use", template.get("default"))
    if not isinstance(template, str) or not template.strip():
        raise ValueError("explicit local tokenizer chat template required")
    return template


def resolve_identity(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"kind", "base_path", "base_revision", "checkpoint_path", "checkpoint_step"}:
        raise ValueError("invalid explicit model spec")
    if spec["kind"] not in {"base", "checkpoint"} or not isinstance(spec["base_path"], str):
        raise ValueError("invalid kind/base path")
    revision = spec["base_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("base revision must be a pinned 40-hex commit, not main/latest")
    base_path = Path(spec["base_path"]).expanduser().resolve(strict=True)
    base_files = artifact_inventory(base_path)
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required <= base_files.keys() or any(name.startswith("adapter_") for name in base_files):
        raise ValueError("base missing assets or contains an adapter")
    verify_weights(base_path, base_files, "model")
    config = load_json(base_path / "config.json")
    if config.get("model_type") != "qwen2" or config.get("is_encoder_decoder", False):
        raise ValueError("expected a Qwen2 decoder-only model")
    if config.get("quantization_config") is not None:
        raise ValueError("CPU float32 evaluation refuses implicit quantized/GPU loaders")
    # A revision label is provenance metadata, not cryptographic proof of Hub origin.
    if base_path.parent.name == "snapshots":
        if base_path.name != revision:
            raise ValueError("snapshot revision differs from requested revision")
        origin = {"model_id": MODEL_ID, "revision": base_path.name, "evidence": "local-HF-snapshot-path"}
    else:
        metadata = load_json(base_path / "research_model_origin.json")
        if metadata != {"model_id": MODEL_ID, "revision": revision}:
            raise ValueError("local origin metadata mismatch")
        origin = {**metadata, "evidence": "local-origin-metadata-not-Hub-attestation"}
    if config.get("_commit_hash") not in {None, revision}:
        raise ValueError("config revision contradicts origin")
    base_fingerprint = digest(base_files)
    identity = {"version": IDENTITY_VERSION, "kind": spec["kind"], "base_path": str(base_path),
                "origin": origin, "base_files": base_files, "base_fingerprint": base_fingerprint,
                "base_config_hash": digest(config), "chat_template_sha256": hashlib.sha256(local_chat_template(base_path).encode()).hexdigest(),
                "checkpoint_path": None, "checkpoint_step": None,
                "checkpoint_files": None, "checkpoint_fingerprint": None}
    if spec["kind"] == "base":
        if spec["checkpoint_path"] is not None or spec["checkpoint_step"] is not None:
            raise ValueError("base mode cannot load checkpoint/LoRA")
    else:
        if not isinstance(spec["checkpoint_path"], str) or type(spec["checkpoint_step"]) is not int or spec["checkpoint_step"] < 0:
            raise ValueError("checkpoint requires explicit directory and step")
        checkpoint = Path(spec["checkpoint_path"]).expanduser().resolve(strict=True)
        if checkpoint == base_path:
            raise ValueError("checkpoint cannot be the base directory")
        files = artifact_inventory(checkpoint)
        verify_weights(checkpoint, files, "adapter_model")
        adapter = load_json(checkpoint / "adapter_config.json")
        if adapter.get("peft_type") != "LORA" or adapter.get("base_model_name_or_path") not in {MODEL_ID, str(base_path)}:
            raise ValueError("adapter config incompatible with explicit base")
        if adapter.get("revision") not in {None, revision}:
            raise ValueError("adapter base revision mismatch")
        binding = load_json(checkpoint / "research_checkpoint.json")
        expected = {"step": spec["checkpoint_step"], "base_fingerprint": base_fingerprint, "base_revision": revision}
        if binding != expected or type(binding["step"]) is not int:
            raise ValueError("checkpoint step/base byte binding mismatch")
        directory_step = re.fullmatch(r"(?:checkpoint-|step_?)?(\d+)", checkpoint.name)
        if directory_step and int(directory_step[1]) != binding["step"]:
            raise ValueError("checkpoint directory step mismatch")
        if "trainer_state.json" in files:
            step = load_json(checkpoint / "trainer_state.json").get("global_step")
            if type(step) is not int or step != binding["step"]:
                raise ValueError("trainer state step mismatch")
        for name in files:
            tokenizer_asset = name.startswith("tokenizer") or name in {"chat_template.jinja", "special_tokens_map.json", "vocab.json", "merges.txt"}
            if tokenizer_asset and files[name] != base_files.get(name):
                raise ValueError("adapter cannot change base tokenizer/template")
        identity.update(checkpoint_path=str(checkpoint), checkpoint_step=binding["step"], checkpoint_files=files,
                        checkpoint_fingerprint=digest(files))
    identity["fingerprint"] = digest({k: identity[k] for k in ("kind", "base_fingerprint", "checkpoint_fingerprint", "checkpoint_step")})
    return identity


def verify_receipt(identity: dict, receipt: dict) -> None:
    required = {"backend", "real_model_loaded", "base_path", "checkpoint_path", "base_fingerprint", "checkpoint_fingerprint",
                "checkpoint_step", "base_config_hash", "active_adapters", "chat_template_sha256", "model_class", "device", "dtype", "versions", "runtime_config_sha256"}
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("invalid loader receipt")
    for key in ("base_path", "checkpoint_path", "base_fingerprint", "checkpoint_fingerprint", "checkpoint_step", "base_config_hash", "chat_template_sha256"):
        if receipt[key] != identity[key]:
            raise ValueError(f"actual loader receipt mismatch: {key}")
    adapters = [] if identity["kind"] == "base" else ["research_eval"]
    if receipt["active_adapters"] != adapters or receipt["device"] != "cpu" or receipt["dtype"] != "float32":
        raise ValueError("unexpected active adapter/device/dtype")
    if not isinstance(receipt["chat_template_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", receipt["chat_template_sha256"]):
        raise ValueError("missing effective chat template hash")
    if receipt["backend"] not in {"transformers-cpu", "cpu-fixture"}:
        raise ValueError("unknown backend receipt")
    if type(receipt["real_model_loaded"]) is not bool or receipt["real_model_loaded"] != (receipt["backend"] == "transformers-cpu"):
        raise ValueError("loader/fixture real-weight attestation mismatch")
    digest(receipt)


def verify_unchanged(spec: dict, identity: dict) -> None:
    if resolve_identity(spec) != identity:
        raise ValueError("model bytes/identity changed after preflight/load/evaluation")

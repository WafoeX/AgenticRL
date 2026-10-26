"""Bounded SFT compatibility pilot. Heavy libraries are lazy; no ART/GRPO path."""

import hashlib
import importlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from .research_environment import TOOL_SCHEMAS
from .research_model_identity import resolve_identity, verify_unchanged
from .research_registration import load_registered_development
from .research_schema import SYSTEM_PROMPT, digest, load_json
from .research_sft import build_expert, pad_batch, tokenize_expert

CONFIG_PATH = "configs/research/sft_pilot_v1.json"
CONFIG = {
    "version": "research-sft-pilot-v1", "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "base_revision": "7ae557604adf67be50417f59c2c2f167def9a775", "seed": 20261011,
    "max_length": 4096, "selection": "first-train-record-per-depth-1-2-3", "updates": 2,
    "batch_size": 3, "learning_rate": 0.0001, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "dtype": "bfloat16", "attention": "eager", "max_wall_seconds": 600,
    "reload_logit_atol": 0.02, "reload_logit_rtol": 0.002,
    "purpose": "compatibility-only-not-formal-SFT-not-checkpoint-selection",
}


def validate_config(config: dict) -> None:
    # Deliberately not a tunable training CLI. A larger run needs a new contract.
    if config != CONFIG or any(type(config.get(k)) is not type(v) for k, v in CONFIG.items()):
        raise ValueError("pilot configuration differs from bounded v1 contract")


def select_records(records: list[dict]) -> list[dict]:
    if not records or any(r.get("split") != "train" for r in records):
        raise ValueError("train only; validation/test forbidden")
    result = []
    for depth in (1, 2, 3):
        candidates = [r for r in records if r["reference_depth"] == depth]
        if not candidates:
            raise ValueError("missing train depth")
        result.append(candidates[0])
    if len({r["sample_id"] for r in result}) != 3:
        raise ValueError("duplicate pilot record")
    return result


def check_budget(remaining_hours: float, allocated_at: str, *, now: datetime | None = None) -> dict:
    if (isinstance(remaining_hours, bool) or not isinstance(remaining_hours, (int, float))
            or not math.isfinite(remaining_hours) or not 0 < remaining_hours <= 20):
        raise ValueError("explicit remaining GPU hours in (0,20] required")
    started = datetime.fromisoformat(allocated_at.replace("Z", "+00:00"))
    if started.tzinfo is None:
        raise ValueError("GPU allocation timestamp requires timezone")
    current = now or datetime.now(timezone.utc)
    elapsed = (current - started).total_seconds()
    if elapsed < 0 or elapsed + CONFIG["max_wall_seconds"] + 60 > remaining_hours * 3600:
        raise ValueError("insufficient remaining budget including allocation/setup and cleanup reserve")
    return {"remaining_hours_at_allocation": remaining_hours, "allocated_at": started.isoformat(),
            "allocated_seconds_before_pilot": elapsed, "pilot_seconds_cap": CONFIG["max_wall_seconds"],
            "cleanup_reserve_seconds": 60}


def validate_batch(batch: dict, rows: list[dict]) -> int:
    if set(batch) != {"input_ids", "attention_mask", "labels"} or len(rows) != 3:
        raise ValueError("pilot batch must contain three complete rows")
    if any(len(batch[k]) != 3 for k in batch):
        raise ValueError("invalid batch cardinality")
    total = 0
    for i, row in enumerate(rows):
        ids, mask, labels = (batch[k][i] for k in ("input_ids", "attention_mask", "labels"))
        n = row["token_count"]
        if (len(ids) != len(mask) or len(ids) != len(labels) or ids[:n] != row["input_ids"]
                or labels[:n] != row["labels"] or mask != [1] * n + [0] * (len(ids) - n)
                or any(label != -100 for label in labels[n:]) or labels[0] != -100):
            raise ValueError("collator changed labels/context/padding")
        supervised = sum(label != -100 for label in labels[1:])
        if not supervised or any(label != -100 and label != token for token, label in zip(ids, labels, strict=True)):
            raise ValueError("empty or invalid causal targets")
        total += supervised
    return total


def prepare(root: Path, base: Path) -> tuple[dict, list[dict], list[dict], dict]:
    config = load_json(root / CONFIG_PATH)
    validate_config(config)
    records, source = load_registered_development(root, split="train")
    selected = select_records(records)
    spec = {"kind": "base", "base_path": str(base), "base_revision": config["base_revision"],
            "checkpoint_path": None, "checkpoint_step": None}
    identity = resolve_identity(spec)
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(identity["base_path"], local_files_only=True,
                                                            trust_remote_code=False, use_fast=True)
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != identity["chat_template_sha256"]:
        raise ValueError("effective tokenizer differs from source identity")
    experts = [build_expert(r) for r in selected]
    rows = [tokenize_expert(e, tokenizer, max_length=config["max_length"]) for e in experts]
    batch = pad_batch(rows, pad_token_id=tokenizer.pad_token_id)
    supervised = validate_batch(batch, rows)
    receipt = {"config": config, "config_sha256": digest(config), "source": source,
               "identity": identity, "selected_sample_ids": [r["sample_id"] for r in selected],
               "depths": [r["reference_depth"] for r in selected],
               "token_counts": [r["token_count"] for r in rows], "supervised_tokens_per_update": supervised,
               "template_sha256": template_hash, "system_prompt_sha256": digest(SYSTEM_PROMPT),
               "tool_schema_sha256": digest(TOOL_SCHEMAS), "training_performed": False}
    verify_unchanged(spec, identity)
    return receipt, experts, rows, batch


def tensor_fingerprint(tensors: dict, torch) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train_updates(model, batch: dict, config: dict, torch, *, deadline: float, emit) -> dict:
    """Testable optimizer contract, preserving supplied labels without a Trainer collator."""
    validate_config(config)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not parameters or any("lora_" not in n for n in parameters):
        raise ValueError("only nonempty LoRA parameters may train")
    before = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    model.train()
    for step in range(1, config["updates"] + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("pilot wall deadline reached before update")
        step_before = {n: p.detach().cpu().clone() for n, p in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False)
        loss = output.loss
        targets = batch["labels"][:, 1:]
        count = (targets != -100).sum().item()
        if not count:
            raise ValueError("empty shifted supervision")
        # Audit HF causal loss independently: context/observation/padding cannot contribute.
        manual = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                                                   targets.reshape(-1), ignore_index=-100)
        if (not torch.isfinite(loss).item() or loss.item() <= 0
                or not torch.allclose(loss.float(), manual.float(), atol=1e-4, rtol=1e-4)):
            raise ValueError("nonfinite/empty or mask-inconsistent loss")
        loss.backward()
        gradients = [p.grad for p in parameters.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
            raise ValueError("missing/nonfinite gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters.values(), config["max_grad_norm"])
        if not math.isfinite(norm.item()) or norm.item() <= 0:
            raise ValueError("zero/nonfinite gradient norm")
        optimizer.step()
        changed = sum(not torch.equal(before[n], p.detach().cpu()) for n, p in parameters.items())
        changed_this_step = sum(not torch.equal(step_before[n], p.detach().cpu()) for n, p in parameters.items())
        if not changed_this_step:
            raise ValueError("optimizer did not update weights this step")
        emit({"step": step, "loss": loss.item(), "manual_masked_loss": manual.item(),
              "shifted_supervised_tokens": count, "grad_norm_before_clip": norm.item(),
              "changed_tensors_from_initial": changed, "changed_tensors_this_update": changed_this_step})
    after = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    return {"actual_updates": config["updates"], "trainable_parameters": sum(p.numel() for p in parameters.values()),
            "trainable_tensors": len(parameters), "initial_adapter_sha256": tensor_fingerprint(before, torch),
            "final_adapter_sha256": tensor_fingerprint(after, torch),
            "max_parameter_delta": max((after[n].float() - before[n].float()).abs().max().item() for n in before)}

"""R4 GPU-pilot evidence helpers. Importing this module never imports Torch/ART."""
import copy
import math
import re
from collections import Counter
from pathlib import Path

from .research_grpo import (
    InfraFailure,
    collect_group,
    member_seed,
    run_steps,
    validate_config,
)
from .research_grpo_runtime import write_json
from .research_model_identity import file_hash, resolve_identity

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SELECTED_FINGERPRINT = "bff2aff07cb640cd240f2fc26e331b0b4c9859af2b4681d45ac8c1c58d7e4557"
BASE_FINGERPRINT = "5572b0e15dbd0cc2e6cc4f23aac1bab454184c1ad2170cd5bcd64230e535864c"
TEMPLATE_SHA = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"


def pilot_config(config):
    c = validate_config(config)
    # This entry is a bounded diagnostic, not a general training launcher.
    if (c["planned_steps"] != 2 or c["groups_per_step"] != 2 or c["group_size"] != 4
            or c["learning_rate"] != 5e-6 or c["beta"] != 0
            or c["max_turns"] != 6 or c["max_completion_tokens"] != 512
            or c["max_total_completion_tokens"] != 1536 or c["max_prompt_tokens"] != 3584):
        raise ValueError("pilot envelope is fixed; new budget requires a new entry/version")
    return c


def selected_identity(base, adapter):
    spec = {"kind": "checkpoint", "base_path": str(Path(base).absolute()), "base_revision": REVISION,
            "checkpoint_path": str(Path(adapter).absolute()), "checkpoint_step": 10}
    identity = resolve_identity(spec)
    if (identity["fingerprint"] != SELECTED_FINGERPRINT or identity["base_fingerprint"] != BASE_FINGERPRINT
            or identity["chat_template_sha256"] != TEMPLATE_SHA):
        raise ValueError("source is not the previously reviewed selected SFT step10")
    return spec, identity


def policy_pair(token):
    match = re.fullmatch(r"token_id:(\d+)", token.token)
    if match is None:
        raise ValueError("raw Choice has no valid sampled policy token ID")
    try:
        token_id, logprob = int(match[1]), float(token.logprob)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("raw Choice has invalid numeric policy payload") from exc
    if not math.isfinite(logprob) or logprob > 0:
        raise ValueError("raw Choice policy logprob must be finite and nonpositive")
    return token_id, logprob


def check_packed(tensors, groups):
    """Audit the actual backend pack, including ART's zero-advantage exclusion.

    Packing may shuffle/reuse prompt prefixes, but may not drop policy tokens.
    Old logprobs are float32 in ART, hence explicit numerical tolerance.
    """
    expected = []
    excluded = 0
    for group in groups:
        mean = sum(t.reward for t in group) / len(group)
        for trajectory in group:
            policy = [policy_pair(t) for choice in trajectory.messages_and_choices if not isinstance(choice, dict)
                      for t in choice.logprobs.content]
            if trajectory.reward == mean:
                excluded += len(policy)
            else:
                expected.extend(policy)
    mask = tensors["assistant_mask"].bool()
    tokens = tensors["tokens"][mask].tolist()
    lps = tensors["logprobs"][mask].tolist()
    if Counter(tokens) != Counter(t for t, _ in expected):
        raise ValueError("ART pack dropped/duplicated/changed policy token IDs")
    actual = sorted(zip(tokens, lps, strict=True))
    reference = sorted(expected)
    if any(not math.isfinite(lp) or not math.isclose(lp, old, abs_tol=2e-5, rel_tol=2e-6)
           for (_, lp), (_, old) in zip(actual, reference, strict=True)):
        raise ValueError("packed old logprobs differ from raw sampled Choice")
    if not bool(tensors["logprobs"][~mask].isnan().all()):
        raise ValueError("nonassistant/padding tokens have policy logprobs")
    if not expected:
        raise ValueError("pack contains no trainable policy signal")
    return {"policy_tokens": len(expected), "zero_advantage_policy_tokens_excluded_by_ART": excluded,
            "shape": list(tensors["tokens"].shape), "raw_policy_ids_equal": True,
            "raw_logprobs_equal_float32_tolerance": True, "nonassistant_logprobs_nan": True}


async def failure_probe(record, config, *, rollout, log, arm="G1"):
    """Explicit synthetic infrastructure injection, with real sibling generations.

    Retry probe: first member fails once, all other members and full retry are real.
    Skip probe: no model requests, all members fail twice. Neither enters training.
    """
    c = copy.deepcopy(config)
    calls = 0
    async def fail_once(row, seed):
        nonlocal calls
        calls += 1
        if calls == 1:
            from .research_grpo import TrainingRollout
            return TrainingRollout(reward=None, termination="infra_failure", infra_error="CONTROLLED injected pre-request outage")
        return await rollout(row, seed)
    def marked(event):
        log({**event, "controlled_probe": True, "counts_as_learning": False})
    retained = await collect_group(record, c, 10000, 0, rollout=fail_once, log=marked)
    if retained is None:
        raise InfraFailure("controlled retry's real generations failed; probe incomplete")
    async def always_fail(row, seed):
        from .research_grpo import TrainingRollout
        return TrainingRollout(reward=None, termination="infra_failure", infra_error="CONTROLLED injected outage both attempts")
    skipped = await collect_group(record, c, 10001, 0, rollout=always_fail, log=marked)
    if skipped is not None or calls != 2 * c["group_size"]:
        raise ValueError("whole-group retry/skip evidence mismatch")
    # Controlled zero-variance guard on copies, never relabel live training rewards.
    zero_config = {**c, "planned_steps": 1, "groups_per_step": 1}
    iterator = iter(retained)
    async def copied_zero(row, seed):
        member = copy.deepcopy(next(iterator))
        member.reward = 0.0
        member.reward_breakdown = {"CONTROLLED_zero_variance": True}
        return member
    async def forbidden_update(*args, **kwargs):
        raise AssertionError("zero variance must not call actual backend")
    zero = await run_steps([record], zero_config, arm=arm, rollout=copied_zero,
                           update=forbidden_update, log=lambda event: marked({**event, "kind": "controlled_zero_" + event["kind"]}))
    if zero["backend_calls"] != 0 or zero["effective_updates"] != 0:
        raise ValueError("zero-variance guard failed")
    return {"synthetic_infra_injection": True, "real_sibling_rollouts": calls - 1,
            "controlled_zero_variance_backend_calls": 0, "controlled_zero_variance_counts_as_learning": False,
            "retry_full_group_verified": True, "second_failure_skipped_full_group": True,
            "member_seeds": [member_seed(c, 10000, 0, i) for i in range(c["group_size"])],
            "probe_used_for_training": False}


def saved_art_inventory(directory):
    """Audit ART 0.5.17 save_model output, NOT an input-model identity.

    HF Trainer writes training_args.bin even with save_strategy='no'. Hash that
    exact opaque metadata file without loading/unpickling it. Never allow pickle
    model weights, optimizer restoration, arbitrary code or linked artifacts.
    Source checkpoints remain subject to the stricter artifact_inventory.
    """
    directory = Path(directory)
    adapter_names = {"adapter_config.json", "adapter_model.safetensors"}
    metadata_names = {"README.md", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
                      "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "generation_config.json"}
    result = {"adapter": {}, "metadata": {}, "opaque_training_metadata": {},
              "opaque_files_deserialized": False, "optimizer_restored": False}
    for path in sorted(directory.iterdir()):
        if path.name not in adapter_names | metadata_names | {"training_args.bin"}:
            raise ValueError("unexpected ART saved artifact: " + path.name)
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size == 0:
            raise ValueError("linked/empty/non-file ART saved artifact: " + path.name)
        bucket = "adapter" if path.name in adapter_names else "opaque_training_metadata" if path.name == "training_args.bin" else "metadata"
        result[bucket][path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if set(result["adapter"]) != adapter_names:
        raise ValueError("ART save missing required safetensors adapter/config")
    return result


def write_starts(spec, identity, path):
    write_json(path, {"version": "research-shared-sft-starts-v1", "selected": {"step": 10},
                      "source": identity, "arms": {a: {"model_spec": spec, "identity": identity} for a in ("G1", "G2")}})

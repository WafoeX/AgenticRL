"""No-training cache experiment contracts; never authorize R4/R5 from a result."""
import math
import time

from .research_schema import digest

VERSION = "research-lora-cache-diagnostic-v2"
WORKERS = ("on-E1", "on-E2", "off-E1", "off-E2")
MAX_REQUESTS = 72
WALL_CAP = 1200
WORKER_CAP = 360


class DiagnosticOutcome(ValueError):
    """Expected numerical/evidence outcome, not an infrastructure retry signal."""


def cache_enabled(worker):
    if worker not in WORKERS:
        raise ValueError("unknown cache worker")
    return worker.startswith("on-")


def effective_config(actual, enabled):
    value = actual.cache_config.enable_prefix_caching
    if type(value) is not bool or value != enabled:
        raise ValueError("effective prefix caching differs from requested condition")
    chunked = actual.scheduler_config.enable_chunked_prefill
    batch = actual.scheduler_config.max_num_batched_tokens
    if type(chunked) is not bool or type(batch) is not int or batch <= 0:
        raise ValueError("effective prefill configuration missing")
    return {"enable_prefix_caching": value, "enable_chunked_prefill": chunked,
            "max_num_batched_tokens": batch, "block_size": actual.cache_config.block_size}


def compare_configs(left, right):
    if (type(left.get("enable_prefix_caching")) is not bool or not left["enable_prefix_caching"]
            or type(right.get("enable_prefix_caching")) is not bool or right["enable_prefix_caching"]):
        raise ValueError("cache conditions not on/off")
    if set(left) != set(right):
        raise ValueError("effective configuration coverage differs")
    if any(left[k] != right[k] for k in left if k != "enable_prefix_caching"):
        raise ValueError("single-variable contract violated")
    return True


class CacheObserver:
    """Pass-through AsyncLLM.generate observer; no sampling/config/output mutation.

    RequestOutput.num_cached_tokens is optional. Missing fields are UNKNOWN,
    never zero. prompt minus cached is derived, NOT scheduler prefill work.
    Installation is only on this diagnostic's private engine instance.
    """
    def __init__(self, engine):
        self.outputs = []
        self.original = engine.generate
        async def observe(*args, **kwargs):
            async for output in self.original(*args, **kwargs):
                if getattr(output, "finished", False):
                    cached = getattr(output, "num_cached_tokens", None)
                    ids = getattr(output, "prompt_token_ids", None)
                    self.outputs.append({"request_id": str(getattr(output, "request_id", "unknown")),
                        "cached_tokens": cached if type(cached) is int else None,
                        "prompt_tokens": len(ids) if isinstance(ids, list) else None,
                        "prompt_sha256": digest(ids) if isinstance(ids, list) else None,
                        "source": "AsyncLLM.generate final RequestOutput.num_cached_tokens"})
                yield output
        engine.generate = observe

    def receipt(self, start, prompt_length, prompt_hash=None):
        final = self.outputs[start:]
        if len(final) != 1:
            return {"available": False, "reason": "expected one final RequestOutput",
                    "final_output_count": len(final), "scheduler_prefill_tokens": None}
        value = final[0]
        cached = value["cached_tokens"]
        if (value["prompt_tokens"] != prompt_length or cached is None
                or (prompt_hash is not None and value.get("prompt_sha256") != prompt_hash)):
            return {**value, "available": False, "reason": "cache field/prompt identity unavailable",
                    "scheduler_prefill_tokens": None}
        if not 0 <= cached <= prompt_length:
            raise ValueError("invalid RequestOutput cached token count")
        return {**value, "available": True, "uncached_prompt_tokens_derived": prompt_length - cached,
                "scheduler_prefill_tokens": None,
                "scheduler_prefill_reason": "not observed; derived uncached length is not scheduler work"}


def remaining(started, cap=WALL_CAP):
    budget = cap - (time.monotonic() - started)
    if not math.isfinite(budget) or budget <= 0:
        raise TimeoutError("experiment wall cap reached; no next worker/retry")
    return budget


def validate_worker(receipt):
    expected_training = {"step_before": 10, "step_after": 10, "optimizer_state_entries": 0,
                         "trainer_tensors_unchanged": True, "optimizer_step_forbidden": True}
    if receipt.get("no_training_verified") != expected_training:
        raise ValueError("worker no-training contract incomplete")
    if not receipt.get("explicit_engine_shutdown_returned") or receipt.get("GPU_other_processes_after") != []:
        raise ValueError("worker engine cleanup not verified")
    cost = receipt.get("cost", {})
    if (set(cost) != {"requests", "prompt_tokens", "completion_tokens", "usage_unknown_requests"}
            or any(type(v) is not int or v < 0 for v in cost.values())
            or cost.get("requests") != 18 or cost.get("completion_tokens") != 18
            or cost.get("usage_unknown_requests") != 0 or cost.get("prompt_tokens", 0) <= 0
            or len(receipt.get("probes", [])) != 18):
        raise ValueError("worker request/token cost incomplete")


def decision(on, off, telemetry_available):
    # On numerical failure is a result, not infrastructure failure. An off
    # candidate still needs top20/API-path verification and actual R4 review.
    fields = ("adapter_distinguished_from_base_all_prefixes", "resident_fingerprints_same_across_engines")
    if not all(type(x.get(k)) is bool and x[k] for x in (on, off) for k in fields):
        raise ValueError("adapter/loading control failed; do not propose candidate")
    a = on["unchanged_gate_1e_4_all_stability_checks_passed"]
    b = off["unchanged_gate_1e_4_all_stability_checks_passed"]
    if b and not a:
        status = "cache-off-candidate-awaits-top20-review" if telemetry_available else "cache-off-candidate-telemetry-gap"
    elif a and b:
        status = "not-reproduced-no-fix-accepted"
    else:
        status = "cache-off-still-fails-stop"
    return {"status": status, "candidate_only": b and not a, "telemetry_available": telemetry_available,
            "R4_accepted": False, "R5_authorized": False, "training_authorized": False,
            "original_top20_path_verified": False, "root_cause_proven": False}

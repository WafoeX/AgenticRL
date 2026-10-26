"""CPU-importable private-engine isolation; no guessed OpenAI ID protocol.

This changes the request entry to direct engine.generate, NOT an API-path fix.
The ledger covers this engine's generate boundary, not internal GPU forwards.
"""
import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import TypeGuard

from .research_schema import digest

VERSION = "research-lora-isolation-v1"
REQUESTS = 3
MAX_PROMPT_TOKENS = REQUESTS * 4095
WALL_CAP = 900
WORKER_CAP = 780


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def token_ids(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)


def prompt_ids(prompt):
    return prompt.get("prompt_token_ids") if isinstance(prompt, dict) else None


class IsolationLedger:
    """Install on ONE engine before publishing it to a server; never on a class.

    Every attempt is counted. Unplanned/duplicate calls are durably recorded and
    rejected BEFORE invoking the original engine. Overflow invalidates evidence
    and rejects, never silently passes unrecorded traffic. No tensor operations.
    """
    def __init__(self, engine, path, *, max_attempts=16, max_events=8):
        if (type(max_attempts) is not int or type(max_events) is not int
                or not 3 <= max_attempts <= 64 or not 1 <= max_events <= 32):
            raise ValueError("bounded recorder limits required")
        self.engine, self.original, self.path = engine, engine.generate, Path(path)
        self.signature = inspect.signature(self.original)
        if not {"prompt", "sampling_params", "request_id", "lora_request"} <= set(self.signature.parameters):
            raise ValueError("unsupported generate signature; stop before server start")
        self.plans = {}
        self.calls = []
        self.max_attempts, self.max_events = max_attempts, max_events
        self.attempts = self.overflow = 0
        self.io_failed = False
        self.sealed = False
        self.closed = False
        self.installed_ns = time.monotonic_ns()

        async def stream(call, args, kwargs):
            error = None
            iterator = None
            try:
                plan = self.plans[call["request_id"]]
                bound = self.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                current = bound.arguments
                if (self.sealed or self.closed or prompt_ids(current["prompt"]) != plan["ids"]
                        or getattr(current["lora_request"], "lora_int_id", None) != plan["lora_id"]
                        or getattr(current["lora_request"], "lora_path", None) != plan["lora_path"]
                        or any(getattr(current["sampling_params"], k, None) != v for k, v in plan["settings"].items())):
                    raise RuntimeError("authorized arguments mutated or window sealed before consumption")
                call["executed"] = True
                self.save()
                iterator = self.original(*args, **kwargs)
                async for output in iterator:
                    if len(call["events"]) >= self.max_events:
                        call["truncated"] = True
                        self.save()
                        raise RuntimeError("engine event ledger overflow")
                    self.capture(call, output)
                    self.save()
                    yield output
                call["exhausted"] = True
            except BaseException as exc:
                error = exc
                call["exception_type"] = type(exc).__name__
                raise
            finally:
                try:
                    if iterator is not None:
                        await iterator.aclose()
                except BaseException as exc:
                    call["iterator_close_error"] = type(exc).__name__
                    if error is None:
                        error = exc
                        raise
                    error.add_note("iterator close failed: " + type(exc).__name__)
                finally:
                    call["closed"] = True
                    call["ended_ns"] = time.monotonic_ns()
                    self.save(error)

        self._stream = stream
        self.wrapper = self.observe
        self.save()
        engine.generate = self.wrapper

    def observe(self, *args, **kwargs):
        # Capture at invocation, even if the returned iterator is never consumed.
        call = self.begin(args, kwargs)
        if not call["authorized"]:
            call.update(closed=True, ended_ns=time.monotonic_ns(), exception_type="RuntimeError")
            self.save()
            raise RuntimeError("unplanned/duplicate engine invocation blocked")
        return self._stream(call, args, kwargs)

    def authorize(self, request_id, ids, lora, sampling):
        if (self.sealed or self.io_failed or self.overflow or self.attempts != len(self.plans)
                or any(not self.receipt(row)["observation_complete"] for row in self.calls)
                or len(self.plans) >= REQUESTS or request_id in self.plans
                or not isinstance(request_id, str) or not request_id or not token_ids(ids) or not 1 <= len(ids) <= 4095
                or type(getattr(lora, "lora_int_id", None)) is not int or lora.lora_int_id <= 0
                or not isinstance(getattr(lora, "lora_path", None), str)
                or type(sampling.max_tokens) is not int or sampling.max_tokens != 1 or sampling.temperature != 0
                or type(sampling.n) is not int or sampling.n != 1 or type(sampling.logprobs) is not int or sampling.logprobs <= 0
                or type(sampling.seed) is not int or sampling.seed != 20261011
                or type(sampling.ignore_eos) is not bool or not sampling.ignore_eos
                or type(sampling.detokenize) is not bool or sampling.detokenize):
            raise ValueError("invalid/duplicate bounded A plan")
        settings = {k: getattr(sampling, k, None) for k in
                    ("n", "temperature", "max_tokens", "logprobs", "seed", "ignore_eos", "detokenize")}
        self.plans[request_id] = {"ids": list(ids), "lora": lora, "sampling": sampling,
                                  "settings": settings, "lora_path": lora.lora_path,
                                  "lora_id": lora.lora_int_id, "used": False}
        self.save()

    def begin(self, args, kwargs):
        self.attempts += 1
        if len(self.calls) >= self.max_attempts:
            self.overflow += 1
            self.save()
            raise RuntimeError("attempt ledger overflow; invocation NOT executed")
        row = {"call_id": len(self.calls), "started_ns": time.monotonic_ns(),
               "authorized": False, "executed": False, "closed": False, "exhausted": False,
               "exception_type": None, "events": [], "truncated": False,
               "request_id": None, "lora_id": None, "lora_path": None,
               "prompt_tokens": None, "prompt_sha256": None, "binding_error": None}
        self.calls.append(row)
        try:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            rid, lora = values["request_id"], values["lora_request"]
            ids = prompt_ids(values["prompt"])
            row.update(request_id=rid if isinstance(rid, str) else None,
                       lora_id=getattr(lora, "lora_int_id", None), lora_path=getattr(lora, "lora_path", None),
                       prompt_tokens=len(ids) if token_ids(ids) else None,
                       prompt_sha256=digest(ids) if token_ids(ids) else None)
            plan = self.plans.get(rid) if isinstance(rid, str) else None
            if (not self.sealed and plan and not plan["used"] and ids == plan["ids"]
                    and lora is plan["lora"] and row["lora_path"] == plan["lora_path"] and row["lora_id"] == plan["lora_id"]
                    and values["sampling_params"] is plan["sampling"]
                    and all(getattr(values["sampling_params"], k, None) == v for k, v in plan["settings"].items())):
                row["authorized"] = True
                plan["used"] = True
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            row["binding_error"] = type(exc).__name__
        self.save()
        return row

    def capture(self, call, output):
        ids = getattr(output, "prompt_token_ids", None)
        completions = getattr(output, "outputs", None)
        completion_ids = [getattr(x, "token_ids", None) for x in completions] if isinstance(completions, list) else None
        generated = None
        if isinstance(completion_ids, list) and len(completion_ids) == 1 and token_ids(completion_ids[0]):
            generated = list(completion_ids[0])
        rid, finished, cached = (getattr(output, k, None) for k in ("request_id", "finished", "num_cached_tokens"))
        call["events"].append({"sequence": len(call["events"]), "observed_ns": time.monotonic_ns(),
            "object_id_process_local": id(output), "request_id": rid if isinstance(rid, str) else None,
            "finished": finished if type(finished) is bool else None,
            "prompt_tokens": len(ids) if token_ids(ids) else None,
            "prompt_sha256": digest(ids) if token_ids(ids) else None,
            "cached_tokens": cached if type(cached) is int else None,
            "completion_tokens": len(generated) if generated is not None else None,
            "generated_token_ids": generated})

    @staticmethod
    def receipt(row):
        finals = [e for e in row["events"] if type(e["finished"]) is bool and e["finished"]]
        complete = (row["authorized"] and row["executed"] and row["closed"] and row["exhausted"]
                    and not row["exception_type"] and not row.get("iterator_close_error")
                    and not row["truncated"] and len(finals) == 1 and bool(row["events"])
                    and finals[0] is row["events"][-1]
                    and all(e["finished"] is not None and e["request_id"] == row["request_id"]
                            and e["prompt_sha256"] == row["prompt_sha256"]
                            and e["prompt_tokens"] == row["prompt_tokens"] for e in row["events"]))
        final = finals[0] if len(finals) == 1 else {}
        usage_known = bool(complete and final.get("completion_tokens") == 1)
        cached = final.get("cached_tokens")
        return {"call_id": row["call_id"], "request_id": row["request_id"], "final_count": len(finals),
                "usage_known": usage_known,
                "observation_complete": bool(usage_known and type(cached) is int and 0 <= cached <= row["prompt_tokens"]),
                "prompt_tokens": row["prompt_tokens"] if usage_known else None,
                "completion_tokens": 1 if usage_known else None,
                "cached_tokens": cached, "scheduler_prefill_tokens": None}

    def receipt_for(self, request_id):
        matches = [row for row in self.calls if row["request_id"] == request_id]
        if len(matches) != 1:
            raise ValueError("missing/duplicate exact request identity; never choose by time or last output")
        return self.receipt(matches[0])

    def snapshot(self):
        receipts = [self.receipt(row) for row in self.calls]
        executed = sum(row["executed"] for row in self.calls)
        known = [r for r in receipts if r["usage_known"]]
        return {"version": VERSION, "installed_ns": self.installed_ns,
            "scope": "all generate invocations on this private engine from pre-server install through shutdown",
            "signature": str(self.signature), "sealed": self.sealed, "closed": self.closed,
            "attempts": self.attempts, "executed_requests": executed,
            "blocked_attempts": sum(not row["authorized"] for row in self.calls) + self.overflow,
            "authorized_not_executed_attempts": sum(row["authorized"] and not row["executed"] for row in self.calls),
            "overflow_attempts": self.overflow,
            "io_failed": self.io_failed, "calls": copy.deepcopy(self.calls), "receipts": receipts,
            "planned_request_ids": list(self.plans),
            "planned_sampling": copy.deepcopy({k: v["settings"] for k, v in self.plans.items()}),
            "unused_planned_request_ids": [k for k, v in self.plans.items() if not v["used"]],
            "known_prompt_tokens": sum(r["prompt_tokens"] for r in known),
            "known_completion_tokens": len(known), "usage_unknown_executed_requests": executed - len(known),
            "GPU_allocation_hours": None, "GPU_allocation_cost_verified": False}

    def save(self, original_error=None):
        try:
            atomic_json(self.path, self.snapshot())
        except Exception as exc:
            self.io_failed = True
            if original_error is None:
                raise
            original_error.add_note("ledger persistence failed: " + type(exc).__name__)

    def seal(self):
        self.sealed = True
        self.save()

    def finish(self):
        # Keep the wrapper installed: late traffic must still be blocked/recorded.
        if self.engine.generate is not self.wrapper:
            raise RuntimeError("engine observer replaced; completeness unverifiable")
        self.sealed = self.closed = True
        self.save()
        return self.validate()

    def validate(self):
        value = self.snapshot()
        validate_closed_ledger(value)
        return value


def validate_closed_ledger(value):
    """Offline rederive receipts and counts; never trust a serialized pass flag."""
    try:
        calls = value["calls"]
        receipts = [IsolationLedger.receipt(row) for row in calls]
        known = [r for r in receipts if r["usage_known"]]
        if (not value["closed"] or not value["sealed"] or value["io_failed"] or value["overflow_attempts"]
                or value["attempts"] != REQUESTS or value["executed_requests"] != REQUESTS
                or len(calls) != REQUESTS or len(value["planned_request_ids"]) != REQUESTS
                or len(set(value["planned_request_ids"])) != REQUESTS or value["unused_planned_request_ids"]
                or [r["request_id"] for r in calls] != value["planned_request_ids"]
                or len({r["lora_id"] for r in calls}) != 1 or len({r["lora_path"] for r in calls}) != 1
                or len({r["prompt_sha256"] for r in calls}) != 1
                or value["blocked_attempts"] != 0 or value["usage_unknown_executed_requests"] != REQUESTS - len(known)
                or value["known_prompt_tokens"] != sum(r["prompt_tokens"] for r in known)
                or value["known_completion_tokens"] != len(known) or value["receipts"] != receipts
                or not all(r["observation_complete"] for r in receipts)):
            raise ValueError("incomplete/extra engine traffic, metadata or token usage; retain ledger, no acceptance")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed engine ledger") from exc
    return value


class NonGeneratingWatchdog:
    """GET-only health checks, task liveness and bounded timeout; no retries.

    Failure is latched and raced against the diagnostic, never suppressed by
    ART's done callback or used to evict a still-running service.
    """
    def __init__(self, check, server_task, *, interval: float = 10, timeout: float = 5):
        self.check, self.server_task = check, server_task
        self.interval, self.timeout = interval, timeout
        self.checks = 0
        self.error = None
        self.task = None

    async def tick(self):
        if self.server_task.done():
            raise RuntimeError("owned OpenAI server task stopped")
        await asyncio.wait_for(self.check(), self.timeout)
        self.checks += 1

    async def run(self):
        try:
            while True:
                await self.tick()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    async def guard(self, awaitable):
        work = asyncio.ensure_future(awaitable)
        try:
            if self.task is None:
                self.task = asyncio.create_task(self.run())
            done, _ = await asyncio.wait([work, self.task], return_when=asyncio.FIRST_COMPLETED)
            if self.task in done:
                await self.task
                raise RuntimeError("watchdog unexpectedly returned")
            value = await work
            if self.error:
                raise RuntimeError(self.error)
            return value
        finally:
            if not work.done():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


def engine_distribution(output, prompt_length, vocab_size):
    """Validate direct-engine full vocabulary; do not serialize tensors/logits."""
    if not token_ids(output.prompt_token_ids) or len(output.prompt_token_ids) != prompt_length:
        raise ValueError("prompt coverage mismatch")
    if type(output.finished) is not bool or not output.finished or len(output.outputs) != 1:
        raise ValueError("one finished completion required")
    part = output.outputs[0]
    if not token_ids(part.token_ids) or len(part.token_ids) != 1 or len(part.logprobs) != 1:
        raise ValueError("single generated token/full distribution required")
    scores = part.logprobs[0]
    if not isinstance(scores, dict) or set(scores) != set(range(vocab_size)) or any(type(k) is not int for k in scores):
        raise ValueError("incomplete numeric vocabulary coverage")
    values = [scores[k].logprob for k in range(vocab_size)]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x > 1e-6 for x in values):
        raise ValueError("invalid/nonfinite logprobs")
    peak = max(values)
    log_z = peak + math.log(math.fsum(math.exp(x - peak) for x in values))
    if abs(log_z) > 1e-4 or part.token_ids[0] not in scores:
        raise ValueError("not a normalized full distribution")
    return part.token_ids[0], values


def source_receipt(method):
    """Installed source attestation, not proof it matches an unavailable upstream."""
    source = inspect.getsource(method)
    return {"signature": str(inspect.signature(method)), "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source": source, "upstream_bytes_attested": False}

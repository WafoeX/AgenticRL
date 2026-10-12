"""Research-only inference loop: all scheduled attempts retained, no reward/retries."""

import asyncio
import copy
import json
from collections.abc import Callable

from .research_answers import reject_constant, unique_object
from .research_checker import check_messages
from .research_environment import TOOL_SCHEMAS, ResearchEnvironment, canonical_json
from .research_eval_backend import (
    EvaluationBackend,
    ModelProtocolError,
    PromptBudgetExceeded,
    normalize_response,
)
from .research_eval_protocol import plan_attempts, request_seed, validate_protocol
from .research_model_identity import verify_receipt
from .research_schema import model_messages

PRIMARY_ORDER = (
    "infra_failure", "missing_output", "truncated", "model_protocol_error", "tool_execution_error",
    "unit_error", "parse_error", "value_error", "same_turn_dependency", "wrong_arguments_or_route",
    "missing_step", "no_tool", "invalid_trace", "exception",
)


def make_outcome(record: dict, slot: dict, messages: list, requests: list, events: list, termination: str,
                 errors: list[str], *, received: bool = True) -> dict:
    trace = [{key: event[key] for key in ("turn", "name", "arguments", "observation")} for event in events]
    checker = check_messages(record, messages, trace, termination=termination)
    labels = set(errors) | set(checker["trace"]["errors"])
    reason = checker["answer"]["reason"]
    if reason != "correct":
        labels.add(reason)
    primary = next((error for error in PRIMARY_ORDER if error in labels), "none")
    infra = termination == "infra_failure"
    valid = not infra and termination != "missing_output" and any(m.get("role") == "assistant" for m in messages)
    return {**copy.deepcopy(slot), "received": received, "termination": termination, "infra_failure": infra,
            "valid_attempt": valid, "messages": copy.deepcopy(messages), "requests": copy.deepcopy(requests),
            "tool_events": copy.deepcopy(events), "trace": trace, "checker": checker,
            "final_answer": checker["answer"]["parsed"], "errors": sorted(labels), "primary_error": primary,
            "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in requests),
            "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in requests)}


async def evaluate_one(record: dict, slot: dict, protocol: dict, backend: EvaluationBackend) -> dict:
    messages, requests, events = model_messages(record), [], []
    environment = ResearchEnvironment(record["environment_snapshot"])
    errors, seen_ids, tokens = [], set(), 0
    termination = "truncated"
    for turn in range(protocol["max_turns"]):
        remaining = protocol["max_output_tokens_per_attempt"] - tokens
        if remaining <= 0:
            errors.append("output_token_budget")
            break
        decode = protocol[slot["mode"]]
        request = {"attempt_id": slot["attempt_id"], "turn": turn, "messages": copy.deepcopy(messages),
                   "tools": copy.deepcopy(TOOL_SCHEMAS), "temperature": decode["temperature"], "top_p": decode["top_p"],
                   "seed": request_seed(protocol, slot, turn), "max_new_tokens": min(remaining, protocol["max_new_tokens_per_turn"]),
                   "max_prompt_tokens": protocol["max_prompt_tokens"], "timeout_seconds": protocol["request_timeout_seconds"]}
        event = {"request": request, "raw_response": None, "normalized_message": None, "usage": {}, "error": None}
        requests.append(event)
        try:
            response = await asyncio.wait_for(backend.complete(copy.deepcopy(request)), timeout=request["timeout_seconds"])
            canonical_json(response)
            event["raw_response"] = copy.deepcopy(response)
            message, finish, usage = normalize_response(response, turn)
            if usage["completion_tokens"] > request["max_new_tokens"] or usage["prompt_tokens"] > request["max_prompt_tokens"]:
                raise RuntimeError("backend exceeded requested token budget")
            event["usage"] = copy.deepcopy(usage)
            tokens += usage["completion_tokens"]
            event["normalized_message"] = copy.deepcopy(message)
        except PromptBudgetExceeded:
            event["error"] = "prompt_budget"
            errors.append("prompt_token_budget")
            termination = "truncated"
            break
        except ModelProtocolError:
            event["error"] = "model_protocol_error"
            raw = event["raw_response"]
            choice = raw["choices"][0] if isinstance(raw, dict) and raw.get("choices") else {}
            content = choice.get("raw_text", "")
            messages.append({"role": "assistant", "content": content})
            termination = "truncated" if choice.get("finish_reason") == "length" else "exception"
            errors.append("model_protocol_error")
            break
        except TimeoutError:
            event["error"] = "request_timeout"
            errors.append("infra_failure")
            termination = "infra_failure"
            break
        except Exception as exc:
            event["error"] = "backend_exception:" + type(exc).__name__
            errors.append("infra_failure")
            termination = "infra_failure"
            break
        messages.append(message)
        if finish == "length":
            errors.append("generation_length_limit")
            termination = "truncated"
            break
        if finish == "content_filter":
            errors.append("model_refusal")
            termination = "exception"
            break
        calls = message.get("tool_calls", [])
        if not calls:
            # A returned empty model Choice is a model failure, not a lost request.
            termination = "finished"
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append("empty_model_output")
            break
        if len(calls) > protocol["max_calls_per_turn"] or any(c["id"] in seen_ids for c in calls):
            errors.append("model_protocol_error")
            termination = "exception"
            break
        for index, call in enumerate(calls):
            seen_ids.add(call["id"])
            function = call["function"]
            arguments = function["arguments"]
            invalid = False
            try:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, object_pairs_hook=unique_object, parse_constant=reject_constant)
                if not isinstance(arguments, dict):
                    raise ValueError("expected arguments object")
                canonical_json(arguments)
            except (ValueError, TypeError, RecursionError):
                invalid = True
                arguments = {}  # recorded as invalid, never silently treated as a valid call
            observation = "Error: invalid tool arguments JSON" if invalid else environment.execute(function["name"], arguments)
            events.append({"turn": turn, "name": function["name"], "arguments": copy.deepcopy(arguments), "observation": observation,
                           "call_id": call["id"], "call_index": index, "attempt_id": slot["attempt_id"],
                           "raw_arguments": copy.deepcopy(function["arguments"]), "environment_hash": record["environment_hash"],
                           "observation_source": "local-research-executor" if not invalid else "argument-parser-rejection"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
            if invalid:
                errors.extend(["model_protocol_error", "invalid_tool_arguments"])
                termination = "exception"
                break
            if observation.startswith("Error:"):
                errors.append("tool_execution_error")
        if termination == "exception":
            break
    else:
        errors.append("max_turns")
    return make_outcome(record, slot, messages, requests, events, termination, errors)


async def evaluate_records(records: list[dict], protocol: dict, identity: dict, backend: EvaluationBackend,
                           *, on_result: Callable[[dict], None] | None = None) -> tuple[list[dict], list[dict]]:
    validate_protocol(protocol)
    verify_receipt(identity, backend.receipt)
    plan = plan_attempts(records, protocol)
    semaphore = asyncio.Semaphore(protocol["concurrency"])

    async def worker(slot):
        async with semaphore:
            outcome = await evaluate_one(records[slot["scenario_index"]], slot, protocol, backend)
            if on_result is not None:
                on_result(copy.deepcopy(outcome))
            return outcome

    tasks = [asyncio.create_task(worker(slot)) for slot in plan]
    try:
        results = await asyncio.gather(*tasks)
        return plan, results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await backend.close()


def complete_ledger(records: list[dict], plan: list[dict], results: list[dict]) -> list[dict]:
    by_id = {slot["attempt_id"]: slot for slot in plan}
    if len(by_id) != len(plan) or len({r["attempt_id"] for r in results}) != len(results):
        raise ValueError("duplicate planned/returned attempt identity")
    outcomes = {}
    for result in results:
        slot = by_id.get(result["attempt_id"])
        if slot is None or any(result.get(key) != value for key, value in slot.items()):
            raise ValueError("unknown/mismatched returned attempt identity")
        outcomes[result["attempt_id"]] = result
    return [outcomes.get(slot["attempt_id"]) or make_outcome(
        records[slot["scenario_index"]], slot, model_messages(records[slot["scenario_index"]]), [], [], "missing_output", ["missing_output"], received=False,
    ) for slot in plan]


class RecordedBackend:
    """Offline replay of stored raw responses with exact request matching, no inference."""

    def __init__(self, receipt: dict, outcomes: list[dict]):
        self.receipt = copy.deepcopy(receipt)
        self.requests = {r["attempt_id"]: copy.deepcopy(r["requests"]) for r in outcomes}
        self.closed = False

    async def complete(self, request: dict) -> dict:
        events = self.requests[request["attempt_id"]]
        if request["turn"] >= len(events):
            raise RuntimeError("replay ran past stored requests")
        event = events[request["turn"]]
        if request != event["request"]:
            raise RuntimeError("saved request differs from current protocol/messages")
        if event["error"] == "request_timeout":
            raise TimeoutError("recorded timeout")
        if event["error"] == "prompt_budget":
            raise PromptBudgetExceeded("recorded prompt limit")
        if event["raw_response"] is None:
            raise RuntimeError("recorded backend failure")
        return copy.deepcopy(event["raw_response"])

    async def close(self) -> None:
        self.closed = True

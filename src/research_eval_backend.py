"""Evaluation-only local CPU backend and Qwen tool-call codec. Lazy heavy imports."""

import asyncio
import copy
import hashlib
import importlib
import json
import re
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from .research_answers import reject_constant, unique_object
from .research_environment import canonical_json
from .research_model_identity import (
    artifact_inventory,
    verify_receipt,
    verify_unchanged,
)
from .research_schema import digest, load_json


class ModelProtocolError(ValueError):
    pass


class PromptBudgetExceeded(ValueError):
    pass


class EvaluationBackend(Protocol):
    receipt: dict

    async def complete(self, request: dict) -> dict: ...

    async def close(self) -> None: ...


def decode_qwen(text: str, turn: int) -> dict:
    """Recognize complete <tool_call> JSONs; do not repair or extract final answers."""
    if not isinstance(text, str):
        raise ModelProtocolError("completion is not text")
    if "<tool_call>" not in text and "</tool_call>" not in text:
        return {"role": "assistant", "content": text}
    parts = list(re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL))
    if not parts or text.count("<tool_call>") != len(parts) or text.count("</tool_call>") != len(parts):
        raise ModelProtocolError("partial/nested tool-call tags")
    remaining = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    calls = []
    for index, part in enumerate(parts):
        try:
            item = json.loads(part[1], object_pairs_hook=unique_object, parse_constant=reject_constant)
            if not isinstance(item, dict) or set(item) != {"name", "arguments"} or not isinstance(item["name"], str) or not isinstance(item["arguments"], dict):
                raise ValueError("expected name/arguments object")
            canonical_json(item)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ModelProtocolError("invalid tool-call JSON") from exc
        # The textual Qwen codec has no provider call ID; explicit adapter-generated ID.
        calls.append({"id": f"qwen-turn-{turn}-call-{index}", "type": "function",
                      "function": {"name": item["name"], "arguments": item["arguments"]}})
    return {"role": "assistant", "content": remaining or None, "tool_calls": calls}


def normalize_response(response: dict, turn: int) -> tuple[dict, str, dict]:
    """One request => one Choice. Preserve raw response outside this normalized view."""
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
        raise RuntimeError("backend returned empty/multiple/missing choices")
    usage = response.get("usage")
    if not isinstance(usage, dict) or not {"prompt_tokens", "completion_tokens"} <= usage.keys():
        raise RuntimeError("backend omitted token usage")
    if any(type(usage[key]) is not int or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens")):
        raise RuntimeError("invalid token usage")
    choice = response["choices"][0]
    if not isinstance(choice, dict) or choice.get("finish_reason") not in {"stop", "tool_calls", "length", "content_filter"}:
        raise RuntimeError("unsupported/missing finish reason")
    if "raw_text" in choice:
        message = decode_qwen(choice["raw_text"], turn)
    else:
        message = copy.deepcopy(choice.get("message"))
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("backend message is not assistant")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ModelProtocolError("invalid assistant content")
    calls = message.get("tool_calls", [])
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ModelProtocolError("tool_calls is not a list")
    normalized = {"role": "assistant", "content": message.get("content")}
    ids = set()
    for call in calls:
        if (not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]
                or call["id"] in ids or call.get("type") != "function"
                or not isinstance(call.get("function"), dict) or not isinstance(call["function"].get("name"), str)):
            raise ModelProtocolError("malformed/duplicate tool call identity")
        if not isinstance(call["function"].get("arguments"), (str, dict)):
            raise ModelProtocolError("invalid tool argument container")
        ids.add(call["id"])
    if calls:
        normalized["tool_calls"] = calls
    canonical_json(normalized)
    return normalized, choice["finish_reason"], usage


class TransformersCPUBackend:
    """Read-only local base or one explicit PEFT adapter; no ART registration/train."""

    def __init__(self, spec: dict, identity: dict, protocol: dict):
        if protocol["concurrency"] != 1:
            raise ValueError("CPU Transformers uses concurrency=1 to isolate RNG/model state")
        self._torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        base_path = identity["base_path"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
        with self._torch.device("cpu"):
            model = transformers.AutoModelForCausalLM.from_pretrained(
                base_path, local_files_only=True, trust_remote_code=False, torch_dtype=self._torch.float32,
                device_map=None, use_safetensors=True, attn_implementation="eager",
            )
        actual_path = str(Path(model.config._name_or_path).resolve(strict=True))
        actual_config = model.config.to_dict()
        file_config = load_json(Path(actual_path) / "config.json")
        ignored = {"_name_or_path", "_commit_hash", "transformers_version", "torch_dtype"}
        differences = []
        for key, value in file_config.items():
            if key in ignored or (key in actual_config and actual_config[key] == value):
                continue
            # Transformers 4.57 Qwen2Config clears this dormant official value.
            # Require explicit keys/False: missing values or an enabled window are NOT equivalent.
            dormant_window = (
                key == "sliding_window" and type(value) is int and value == 32768
                and file_config.get("model_type") == actual_config.get("model_type") == "qwen2"
                and type(file_config.get("use_sliding_window")) is bool and not file_config["use_sliding_window"]
                and type(actual_config.get("use_sliding_window")) is bool and not actual_config["use_sliding_window"]
                and "sliding_window" in actual_config and actual_config["sliding_window"] is None
            )
            if not dormant_window:
                differences.append(key)
        if differences:
            raise ValueError("loaded model config differs from actual local config: " + ",".join(differences))
        template = tokenizer.get_chat_template(tools=protocol_tools())
        active, checkpoint_path, checkpoint_files = [], None, None
        versions = {"torch": self._torch.__version__, "transformers": transformers.__version__}
        if identity["kind"] == "checkpoint":
            peft = importlib.import_module("peft")
            checkpoint_path = identity["checkpoint_path"]
            with self._torch.device("cpu"):
                model = peft.PeftModel.from_pretrained(model, checkpoint_path, adapter_name="research_eval",
                                                       is_trainable=False, local_files_only=True, use_safetensors=True)
            model.set_adapter("research_eval")
            active = list(model.active_adapters)
            config_path = str(Path(model.peft_config["research_eval"]._name_or_path).resolve(strict=True)) if hasattr(model.peft_config["research_eval"], "_name_or_path") else checkpoint_path
            if config_path != checkpoint_path:
                raise ValueError("loaded adapter path differs from explicit checkpoint")
            checkpoint_files = artifact_inventory(Path(checkpoint_path))
            versions["peft"] = peft.__version__
        elif getattr(model, "peft_config", None):
            raise ValueError("base mode unexpectedly loaded an adapter")
        model.to("cpu")
        model.eval()
        parameters = list(model.parameters())
        if not parameters or any(str(p.device) != "cpu" or (p.is_floating_point() and p.dtype != self._torch.float32) for p in parameters):
            raise ValueError("loaded tensors not CPU float32")
        for parameter in parameters:
            parameter.requires_grad_(False)
        base_files = artifact_inventory(Path(actual_path))
        self.receipt = {"backend": "transformers-cpu", "real_model_loaded": True, "base_path": actual_path,
                        "checkpoint_path": checkpoint_path, "base_fingerprint": digest(base_files),
                        "checkpoint_fingerprint": digest(checkpoint_files) if checkpoint_files is not None else None,
                        "checkpoint_step": identity["checkpoint_step"], "base_config_hash": digest(file_config),
                        "active_adapters": active, "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                        "model_class": type(model).__name__, "device": "cpu", "dtype": "float32", "versions": versions,
                        "runtime_config_sha256": digest(actual_config)}
        verify_receipt(identity, self.receipt)
        verify_unchanged(spec, identity)
        self._model, self._tokenizer, self._template = model, tokenizer, template
        self._lock = asyncio.Lock()
        self._closed = False

    def _generate(self, request: dict) -> dict:
        tokenizer, model, torch = self._tokenizer, self._model, self._torch
        if self._closed or tokenizer is None or model is None:
            raise RuntimeError("backend closed")
        inputs = tokenizer.apply_chat_template(request["messages"], tools=request["tools"], chat_template=self._template,
                                                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens > request["max_prompt_tokens"]:
            raise PromptBudgetExceeded("prompt exceeds fixed token budget")
        generation = {"max_new_tokens": request["max_new_tokens"], "do_sample": request["temperature"] > 0,
                      "num_beams": 1, "num_return_sequences": 1, "return_dict_in_generate": True,
                      "max_time": request["timeout_seconds"], "pad_token_id": tokenizer.eos_token_id,
                      "eos_token_id": tokenizer.eos_token_id}
        if generation["do_sample"]:
            generation.update(temperature=request["temperature"], top_p=request["top_p"], top_k=0)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.default_generator.manual_seed(request["seed"])
            # Explicit GenerationConfig avoids inheriting checkpoint sampling penalties.
            config = importlib.import_module("transformers").GenerationConfig(**generation)
            output = model.generate(**inputs, generation_config=config)
        ids = output.sequences[0, prompt_tokens:].tolist()
        eos = tokenizer.eos_token_id
        stopped = bool(ids) and ids[-1] == eos
        # Strip only the terminal transport EOS; preserve tool tags/other tokens.
        text = tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False)
        return {"choices": [{"raw_text": text, "finish_reason": "stop" if stopped else "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids)},
                "raw_token_ids": ids, "call_id_source": "adapter-generated-from-text-not-provider"}

    async def complete(self, request: dict) -> dict:
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(self._generate, copy.deepcopy(request)))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Even repeated cancellation must not release a live generation thread.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                with suppress(Exception):
                    task.result()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._model = None
            self._tokenizer = None


def protocol_tools() -> list:
    from .research_environment import TOOL_SCHEMAS
    return copy.deepcopy(TOOL_SCHEMAS)

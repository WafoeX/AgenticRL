"""R4 fresh-arm staging and measured optimizer contracts; no GPU imports on load."""
import copy
import importlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .research_environment import TOOL_SCHEMAS
from .research_grpo import (
    InfraFailure,
    art_train_kwargs,
    to_art_groups,
    training_rollout,
    validate_config,
)
from .research_model_identity import artifact_inventory, file_hash, resolve_identity
from .research_schema import digest, load_json


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_arms(starts_path: Path, output: Path, config: dict, *, code_revision: str, data_sha256: str):
    """New local run only. Copy adapters, NEVER SFT optimizer or an old ART state.

    No registration/loading/training; source resolve_identity verifies real inventories.
    """
    config = validate_config(config)
    starts = load_json(starts_path)
    if starts.get("version") != "research-shared-sft-starts-v1" or set(starts["arms"]) != {"G1", "G2"}:
        raise ValueError("selected shared-SFT starts receipt required")
    for value in (code_revision, data_sha256):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
            raise ValueError("code/data hashes required")
    source = starts["source"]
    identities = {}
    for arm, row in starts["arms"].items():
        identities[arm] = resolve_identity(row["model_spec"])
        if (identities[arm]["fingerprint"] != source["fingerprint"]
                or row["identity"]["fingerprint"] != source["fingerprint"]
                or identities[arm]["checkpoint_step"] != starts["selected"]["step"]):
            raise ValueError("shared selected source mismatch")
    output = output.absolute()
    for identity in identities.values():
        origin = Path(identity["checkpoint_path"])
        if output.resolve().is_relative_to(origin) or origin.is_relative_to(output.resolve()):
            raise ValueError("run/source paths must not overlap")
    output.mkdir(parents=True, exist_ok=False)
    run_id = digest([str(output.resolve()), config, source["fingerprint"], code_revision, data_sha256])[:16]
    receipt = {"version": "research-grpo-staging-v1", "run_id": run_id, "config": config,
               "code_revision": code_revision, "data_sha256": data_sha256,
               "source_fingerprint": source["fingerprint"], "arms": {},
               "actual_GRPO_load_verified": False, "GRPO_training_started": False}
    for arm, identity in identities.items():
        workspace = output / arm
        adapter = workspace / "initial-adapter"
        adapter.mkdir(parents=True, exist_ok=False)
        files = identity["checkpoint_files"]
        # Identity permits only audited adapter artifacts; reject unexpected optimizer content.
        if set(files) != {"adapter_config.json", "adapter_model.safetensors", "research_checkpoint.json"}:
            raise ValueError("initial adapter inventory must not contain optimizer/state/extra files")
        for name in files:
            source_file = Path(identity["checkpoint_path"]) / name
            if source_file.is_symlink() or source_file.stat().st_nlink != 1:
                raise ValueError("linked source artifact not allowed")
            shutil.copyfile(source_file, adapter / name)
            if (adapter / name).stat().st_ino == source_file.stat().st_ino:
                raise ValueError("adapter copy aliases source")
        if artifact_inventory(adapter) != files:
            raise ValueError("new arm adapter byte mismatch")
        (workspace / "art-workspace").mkdir(exist_ok=False)
        spec = {**starts["arms"][arm]["model_spec"], "checkpoint_path": str(adapter.resolve())}
        receipt["arms"][arm] = {"arm": arm, "model_name": f"research-{run_id}-{arm}", "project": f"research-{run_id}-{arm}",
                               "reward_version": config["rewards"][arm], "model_spec": spec,
                               "identity": resolve_identity(spec), "workspace": str(workspace.resolve()),
                               "art_path": str((workspace / "art-workspace").resolve()),
                               "must_create_fresh_optimizer": True, "optimizer_initialized": False}
    write_json(output / "run.json", receipt)
    return receipt


def internal_art_config(config):
    c = validate_config(config)
    return {"init_args": {"load_in_4bit": False, "load_in_8bit": False, "max_seq_length": 4096, "trust_remote_code": False},
            "peft_args": {**c["lora"], "random_state": c["seed"]},
            "engine_args": {"dtype": "bfloat16", "max_model_len": 4096, "max_num_seqs": 1,
                            "seed": c["seed"], "max_lora_rank": c["lora"]["r"], "max_logprobs": 20,
                            "gpu_memory_utilization": 0.45, "enforce_eager": True, "enable_sleep_mode": True}}


class ConfiguredRollout:
    """Bind all sampling/reward knobs to one isolated ART model identity."""
    def __init__(self, arm_receipt, model, tokenizer, config, *, complete):
        self.config = validate_config(config)
        if model.name != arm_receipt["model_name"] or model.project != arm_receipt["project"]:
            raise ValueError("rollout model identity differs from isolated arm")
        if arm_receipt["arm"] not in self.config["rewards"] or arm_receipt["reward_version"] != self.config["rewards"][arm_receipt["arm"]]:
            raise ValueError("reward not registered for this arm")
        self.arm, self.model, self.tokenizer, self.complete = arm_receipt, model, tokenizer, complete

    async def __call__(self, record, seed):
        name = self.model.get_inference_name()
        if not name.startswith(self.arm["model_name"] + "@"):
            raise ValueError("served model identity mismatch")
        def count_prompt(messages, tools):
            return len(self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True))
        return await training_rollout(name, record, self.config, self.arm["reward_version"], seed,
                                      complete=self.complete, count_prompt=count_prompt)


def seed_runtime(config, *, torch=None):
    """Seed Python/NumPy packing and the supplied Torch CPU generator.

    No CUDA access here; the future authorized GPU loader must additionally seed
    CUDA before model registration and record that action, not infer it from YAML.
    """
    c = validate_config(config)
    random.seed(c["seed"])
    try:
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(c["seed"])
    if torch is not None:
        torch.random.default_generator.manual_seed(c["seed"])
    return {"seed": c["seed"], "python_seeded": True, "numpy_seeded": numpy is not None,
            "torch_CPU_seeded": torch is not None, "torch_CUDA_seed_verified": False}


def optimizer_parameters(optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]


def verify_fresh_optimizer(optimizer, parameters):
    actual = optimizer_parameters(optimizer)
    if optimizer.state or not actual or len({id(p) for p in actual}) != len(actual) or {id(p) for p in actual} != {id(p) for p in parameters}:
        raise ValueError("optimizer must be fresh and own exactly this arm's trainable parameters")


def verify_arm_isolation(model_a, optimizer_a, model_b, optimizer_b):
    a, b = optimizer_parameters(optimizer_a), optimizer_parameters(optimizer_b)
    if (model_a is model_b or optimizer_a is optimizer_b or {id(p) for p in a} & {id(p) for p in b}
            or {p.data_ptr() for p in a} & {p.data_ptr() for p in b}):
        raise ValueError("G1/G2 share mutable model/optimizer storage")


def optimizer_step_counts(optimizer):
    return [float(s["step"]) for s in optimizer.state.values() if "step" in s]


class ARTTokenizerView:
    """Pinned ART 0.5.17 packing view; native inference template stays unchanged.

    Fix redundant tool-schema wrapping and remove serialized tool_calls ONLY
    from ART's synthetic assistant sentinel slots (raw policy IDs replace these).
    Otherwise Qwen repeats the tool call as unmasked prompt text after raw IDs.
    This stateful view is for serialized packing, never concurrent generation.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._sentinel = None

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def apply_chat_template(self, messages, *, tools=None, **kwargs):
        if tools is not None:
            tools = [t["function"] if isinstance(t.get("function"), dict)
                     and t["function"].get("type") == "function" and "function" in t["function"] else t
                     for t in tools]
            if tools != TOOL_SCHEMAS:
                raise ValueError("ART packing tools differ from frozen research schemas")
            # Pydantic reorders TypedDict keys; Qwen serializes insertion order.
            # Preserve the exact public prompt, not merely schema equality.
            tools = copy.deepcopy(TOOL_SCHEMAS)
        messages = copy.deepcopy(messages)
        packing = kwargs.get("continue_final_message", False)
        tokenize = kwargs.get("tokenize", True)
        if packing and not tokenize:
            self._sentinel = None  # ART's initial rendered-history call
        replacing = packing and tokenize and self._sentinel is not None
        for message in messages:
            if replacing and message.get("role") == "assistant" and message.get("content") == self._sentinel:
                message.pop("tool_calls", None)
            else:
                for call in message.get("tool_calls", []) or []:
                    fn = call.get("function", call)
                    raw = fn.get("arguments")
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw)
                        except ValueError:
                            continue  # malformed model output stays raw, never repaired
                        if isinstance(args, dict):
                            fn["arguments"] = args
        result = self.tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
        if packing and tokenize:
            if replacing:
                self._sentinel = None
            else:
                sentinel_id = max(set(range(self.tokenizer.vocab_size)) - set(result))
                self._sentinel = self.tokenizer.decode(sentinel_id)
        return result


class MeasuredARTUpdate:
    """Pinned ART bridge AFTER explicit verified GPU loading (not an entry point).

    Supply the actual loaded service trainer/optimizer and verify initial weights
    separately. Never initialize or restore from a different arm here.
    """
    def __init__(self, backend, model, optimizer, parameters, art, *, version=None, tokenizer=None):
        version = version or importlib.metadata.version("openpipe-art")
        if version != "0.5.17":
            raise ValueError("private trainer evidence adapter pinned to ART 0.5.17")
        verify_fresh_optimizer(optimizer, parameters)
        if hasattr(backend, "_tokenizers"):
            if tokenizer is None:
                raise ValueError("explicit verified native tokenizer required for ART packing")
            backend._tokenizers[model.base_model] = ARTTokenizerView(tokenizer)
        self.backend, self.model, self.optimizer, self.art = backend, model, optimizer, art
        self.parameters = list(parameters)

    async def __call__(self, groups, **kwargs):
        before = [p.detach().cpu().clone() for p in self.parameters]
        counts = optimizer_step_counts(self.optimizer)
        previous_step = await self.model.get_step()
        result = await self.backend.train(self.model, to_art_groups(groups, self.art), **kwargs)
        after_counts = optimizer_step_counts(self.optimizer)
        delta = max(after_counts, default=0) - max(counts, default=0)
        changed = sum(not bool((p.detach().cpu() == old).all()) for p, old in zip(self.parameters, before, strict=True))
        lrs = {g["lr"] for g in self.optimizer.param_groups}
        metrics = result.metrics
        if lrs != {kwargs["learning_rate"]}:
            raise ValueError("actual optimizer LR differs from requested LR")
        if delta < 0 or delta != int(delta) or (changed and delta == 0):
            raise ValueError("optimizer update evidence inconsistent")
        if result.step <= previous_step:
            raise ValueError("ART did not advance saved step")
        # Missing metrics are evidence failures, never substitute fake zeros.
        return {"optimizer_steps": int(delta), "changed_tensors": changed,
                "loss": metrics["loss/train"], "grad_norm": metrics["loss/grad_norm"],
                "learning_rate": next(iter(lrs)), "beta": kwargs["kl_penalty_coef"],
                "checkpoint_step": result.step, "checkpoint_path": result.checkpoint_path,
                "ART_metrics": metrics, "previous_checkpoint_step": previous_step}


@contextmanager
def _audit_file(path):
    with Path(path).open("x", encoding="utf-8") as handle:
        yield handle


class AuditLog:
    """Append-only, fsynced raw JSONL; cost includes rejected/retried groups."""
    def __init__(self, path):
        self._stack = ExitStack()
        self.handle = self._stack.enter_context(_audit_file(path))
        self.cost = {"requested_rollouts": 0, "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                     "usage_unknown_requests": 0, "retries": 0, "skipped_groups": 0}
        self.begun = time.monotonic()

    def __call__(self, event):
        row = copy.deepcopy(event)
        if row["kind"] == "rollout":
            self.cost["requested_rollouts"] += 1
            for k in ("requests", "prompt_tokens", "completion_tokens", "usage_unknown_requests"):
                self.cost[k] += row["payload"][k]
        if row["kind"] == "group":
            self.cost["retries"] += row["retry"]
            self.cost["skipped_groups"] += row["skipped"]
        if row["kind"] == "summary":
            row["cost"] = {**self.cost, "known_total_tokens": self.cost["prompt_tokens"] + self.cost["completion_tokens"],
                           "token_cost_complete": self.cost["usage_unknown_requests"] == 0,
                           "program_wall_hours": (time.monotonic() - self.begun) / 3600,
                           "GPU_allocation_hours": None, "GPU_allocation_cost_verified": False}
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self._stack.close()


def openai_transport(client):
    """Require caller to construct AsyncOpenAI(max_retries=0). No borrowed retries."""
    openai = importlib.import_module("openai")
    if client.max_retries != 0:
        raise ValueError("hidden per-request retries prohibited")
    async def complete(**kwargs):
        try:
            return await client.chat.completions.create(**kwargs)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError) as exc:
            raise InfraFailure(type(exc).__name__) from exc
        except openai.BadRequestError as exc:
            if "Already borrowed" not in str(exc):
                raise  # bad configuration is NOT model error and is NOT silently skipped
            raise InfraFailure("vLLM Already borrowed") from exc
    return complete


def implementation_hashes():
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in ("research_grpo.py", "research_grpo_runtime.py", "research_grpo_reward.py", "rewards.py", "reward_v2.py", "research_checker.py", "research_environment.py")}


def effective_contract(config):
    c = validate_config(config)
    return {"config": c, "model_internal_config": internal_art_config(c), "train_kwargs": art_train_kwargs(c),
            "implementation_sha256": implementation_hashes(), "actual_ART_GPU_verified": False}

"""Exercise production CPU adapter with stub libraries, NOT actual model loading."""

import asyncio
import json
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.verify_research_eval_cpu import create_fixture_spec
from src import research_eval_backend as module
from src.research_eval_backend import (
    PromptBudgetExceeded,
    TransformersCPUBackend,
    normalize_response,
)
from src.research_model_identity import resolve_identity
from src.research_schema import load_json

ROOT = Path(__file__).resolve().parents[1]


class TokenRow(list):
    def tolist(self):
        return list(self)


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        row, columns = index
        return TokenRow(self.rows[row][columns])


class Parameter:
    device = "cpu"
    dtype = "float32"
    trainable = True

    def is_floating_point(self):
        return True

    def requires_grad_(self, value):
        self.trainable = value


class FakeModel:
    def __init__(self, config, *, implicit_adapter=False):
        self.config = SimpleNamespace(**config, to_dict=lambda: dict(config))
        self.parameter = Parameter()
        self.eval_called = False
        self.active_adapters = []
        self.generated = []
        self.tokens = [71, 72, 99]
        self.started = threading.Event()
        self.release = threading.Event()
        self.blocked = False
        self.fail = False
        if implicit_adapter:
            self.peft_config = {"old_lora": SimpleNamespace()}

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        self.eval_called = True

    def parameters(self):
        return iter([self.parameter])

    def set_adapter(self, name):
        self.active_adapters = [name]

    def generate(self, **kwargs):
        self.started.set()
        if self.blocked:
            self.release.wait(2)
        if self.fail:
            raise RuntimeError("stub generation failure")
        self.generated.append(kwargs)
        return SimpleNamespace(sequences=Tensor([[1, 2, 3] + self.tokens]))


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, template):
        self.template = template
        self.applied = []
        self.decoded = []

    def get_chat_template(self, **kwargs):
        return self.template

    def apply_chat_template(self, messages, **kwargs):
        self.applied.append((messages, kwargs))
        return {"input_ids": Tensor([[1, 2, 3]]), "attention_mask": Tensor([[1, 1, 1]])}

    def decode(self, ids, **kwargs):
        self.decoded.append((ids, kwargs))
        return '{"value":4,"unit":"dimensionless"}'


def install_stubs(monkeypatch, identity, *, mutate=None):
    log: dict = {"seeds": [], "cpu_contexts": [], "fork_devices": []}
    config = load_json(Path(identity["base_path"]) / "config.json")
    config["_name_or_path"] = identity["base_path"]
    if mutate == "config":
        config["model_type"] = "wrong"
    model = FakeModel(config, implicit_adapter=mutate == "old_lora")
    if mutate == "device":
        model.parameter.device = "cuda:0"
    if mutate == "dtype":
        model.parameter.dtype = "float16"
    template = load_json(Path(identity["base_path"]) / "tokenizer_config.json")["chat_template"]
    tokenizer = FakeTokenizer("wrong" if mutate == "template" else template)

    def tokenizer_loader(path, **kwargs):
        log["tokenizer_load"] = (path, kwargs)
        return tokenizer

    def model_loader(path, **kwargs):
        log["model_load"] = (path, kwargs)
        return model

    def adapter_loader(base, path, **kwargs):
        assert base is model
        log["adapter_load"] = (path, kwargs)
        model.peft_config = {"research_eval": SimpleNamespace(_name_or_path=path)}
        return model

    @contextmanager
    def device_context(device):
        log["cpu_contexts"].append(device)
        yield

    def fork_rng(*, devices):
        log["fork_devices"].append(devices)
        return nullcontext()

    torch = SimpleNamespace(__version__="STUB-NOT-TORCH", float32="float32", device=device_context,
                            random=SimpleNamespace(fork_rng=fork_rng), inference_mode=nullcontext,
                            default_generator=SimpleNamespace(manual_seed=lambda seed: log["seeds"].append(seed)))
    transformers = SimpleNamespace(__version__="STUB-NOT-TRANSFORMERS", AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
                                   AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader), GenerationConfig=SimpleNamespace)
    peft = SimpleNamespace(__version__="STUB-NOT-PEFT", PeftModel=SimpleNamespace(from_pretrained=adapter_loader))
    original = module.importlib.import_module
    libraries = {"torch": torch, "transformers": transformers, "peft": peft}

    def import_module(name, *args):
        return libraries[name] if name in libraries else original(name, *args)

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    return model, tokenizer, log


@pytest.mark.parametrize("checkpoint", [False, True])
def test_explicit_cpu_load_adapter_and_fixed_generation_kwargs(tmp_path, monkeypatch, checkpoint):
    spec = create_fixture_spec(tmp_path, checkpoint=checkpoint)
    identity = resolve_identity(spec)
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    model, tokenizer, log = install_stubs(monkeypatch, identity)
    backend = TransformersCPUBackend(spec, identity, protocol)
    assert model.eval_called and not model.parameter.trainable
    assert log["cpu_contexts"] == (["cpu", "cpu"] if checkpoint else ["cpu"])
    assert log["model_load"] == (identity["base_path"], {"local_files_only": True, "trust_remote_code": False,
                                                         "torch_dtype": "float32", "device_map": None,
                                                         "use_safetensors": True, "attn_implementation": "eager"})
    assert backend.receipt["active_adapters"] == (["research_eval"] if checkpoint else [])
    if checkpoint:
        assert log["adapter_load"] == (identity["checkpoint_path"], {"adapter_name": "research_eval", "is_trainable": False,
                                                                     "local_files_only": True, "use_safetensors": True})
    request = {"messages": [{"role": "user", "content": "2+2"}], "tools": [], "max_prompt_tokens": 10,
               "max_new_tokens": 10, "temperature": 0, "top_p": 1, "seed": 123, "timeout_seconds": 60}
    result = asyncio.run(backend.complete(request))
    message, finish, usage = normalize_response(result, 0)
    assert finish == "stop" and usage == {"prompt_tokens": 3, "completion_tokens": 3}
    assert result["raw_token_ids"] == [71, 72, 99] and result["call_id_source"].startswith("adapter-generated")
    assert message["content"] == '{"value":4,"unit":"dimensionless"}'
    generation = vars(model.generated[0]["generation_config"])
    assert generation == {"max_new_tokens": 10, "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                          "return_dict_in_generate": True, "max_time": 60, "pad_token_id": 99, "eos_token_id": 99}
    request.update(temperature=1.2, top_p=.95, seed=124)
    asyncio.run(backend.complete(request))
    generation = vars(model.generated[-1]["generation_config"])
    assert generation["do_sample"] and generation["top_k"] == 0 and generation["temperature"] == 1.2 and generation["top_p"] == .95
    assert tokenizer.decoded == [([71, 72], {"skip_special_tokens": False})] * 2
    assert log["seeds"] == [123, 124] and log["fork_devices"] == [[], []]
    assert tokenizer.applied[0][1]["chat_template"] == tokenizer.template
    assert tokenizer.applied[0][1]["return_tensors"] == "pt"
    request["max_prompt_tokens"] = 2
    with pytest.raises(PromptBudgetExceeded):
        asyncio.run(backend.complete(request))
    assert len(model.generated) == 2
    request["max_prompt_tokens"] = 10
    model.tokens = [71, 72]
    assert asyncio.run(backend.complete(request))["choices"][0]["finish_reason"] == "length"
    assert tokenizer.decoded[-1][0] == [71, 72]
    asyncio.run(backend.close())
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(backend.complete(request))


@pytest.mark.parametrize("mutate", ["config", "old_lora", "device", "dtype", "template"])
def test_runtime_mismatch_not_just_requested_identity(tmp_path, monkeypatch, mutate):
    spec = create_fixture_spec(tmp_path)
    identity = resolve_identity(spec)
    install_stubs(monkeypatch, identity, mutate=mutate)
    with pytest.raises(ValueError):
        TransformersCPUBackend(spec, identity, load_json(ROOT / "configs/research/evaluation_v1.json"))


@pytest.mark.parametrize("case", [
    "normalized", "unchanged", "enabled", "runtime_enabled", "missing_runtime_window",
    "missing_runtime_flag", "other_window", "other_field", "missing_file_flag", "non_boolean_flag",
])
def test_dormant_qwen_window_normalization_is_narrow(tmp_path, monkeypatch, case):
    spec = create_fixture_spec(tmp_path)
    path = Path(spec["base_path"]) / "config.json"
    file_config = load_json(path)
    file_config.update(use_sliding_window=False, sliding_window=32768)
    if case == "enabled":
        file_config["use_sliding_window"] = True
    elif case == "other_window":
        file_config["sliding_window"] = 4096
    elif case == "missing_file_flag":
        del file_config["use_sliding_window"]
    elif case == "non_boolean_flag":
        file_config["use_sliding_window"] = 0
    path.write_text(json.dumps(file_config), encoding="utf-8")
    identity = resolve_identity(spec)
    model, _, _ = install_stubs(monkeypatch, identity)
    actual = model.config.to_dict()
    if case != "unchanged":
        actual["sliding_window"] = None
    if case == "runtime_enabled":
        actual["use_sliding_window"] = True
    elif case == "missing_runtime_window":
        del actual["sliding_window"]
    elif case == "missing_runtime_flag":
        del actual["use_sliding_window"]
    elif case == "other_field":
        actual["hidden_size"] = file_config.get("hidden_size", 0) + 1
        # Include a field in the file so it is subject to the existing identity check.
        file_config["hidden_size"] = actual["hidden_size"] - 1
        path.write_text(json.dumps(file_config), encoding="utf-8")
        identity = resolve_identity(spec)
    model.config.to_dict = lambda: dict(actual)
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    if case in {"normalized", "unchanged"}:
        backend = TransformersCPUBackend(spec, identity, protocol)
        assert backend.receipt["runtime_config_sha256"]
        asyncio.run(backend.close())
    else:
        with pytest.raises(ValueError, match="loaded model config differs"):
            TransformersCPUBackend(spec, identity, protocol)


@pytest.mark.parametrize("repeat_cancel", [False, True])
def test_cancellation_waits_for_stub_thread_then_close(tmp_path, monkeypatch, repeat_cancel):
    spec = create_fixture_spec(tmp_path)
    identity = resolve_identity(spec)
    model, _, _ = install_stubs(monkeypatch, identity)
    model.blocked = True
    backend = TransformersCPUBackend(spec, identity, load_json(ROOT / "configs/research/evaluation_v1.json"))
    request = {"messages": [], "tools": [], "max_prompt_tokens": 10, "max_new_tokens": 10,
               "temperature": 0, "top_p": 1, "seed": 1, "timeout_seconds": 60}

    async def run():
        task = asyncio.create_task(backend.complete(request))
        try:
            for _ in range(1000):
                if model.started.is_set():
                    break
                await asyncio.sleep(.001)
            assert model.started.is_set()
            task.cancel()
            await asyncio.sleep(.005)
            assert not task.done()
            if repeat_cancel:
                task.cancel()
                await asyncio.sleep(.005)
                assert not task.done()
        finally:
            model.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await backend.close()
        assert backend._model is None and backend._tokenizer is None

    asyncio.run(run())

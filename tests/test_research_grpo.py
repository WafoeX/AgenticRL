"""CPU contracts: raw OpenAI Choices, fixed groups, measurable tiny optimizer."""
import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research_grpo import (
    InfraFailure,
    TrainingRollout,
    art_train_kwargs,
    collect_group,
    member_seed,
    run_steps,
    scene_plan,
    training_rollout,
    validate_config,
)
from src.research_grpo_reward import score
from src.research_grpo_runtime import (
    ARTTokenizerView,
    AuditLog,
    ConfiguredRollout,
    MeasuredARTUpdate,
    effective_contract,
    internal_art_config,
    prepare_arms,
    seed_runtime,
    verify_arm_isolation,
    verify_fresh_optimizer,
)
from src.research_schema import load_json

ROOT = Path(__file__).resolve().parents[1]


def _choice_class():
    # Legacy CPU tests put a non-package openai double in sys.modules during
    # collection. Import the real SDK class without changing that test's binding.
    old = sys.modules.get("openai")
    if old is not None and not hasattr(old, "__path__"):
        del sys.modules["openai"]
    try:
        return importlib.import_module("openai.types.chat.chat_completion").Choice
    finally:
        if old is not None:
            sys.modules["openai"] = old


Choice = _choice_class()


@pytest.fixture
def config():
    return load_json(ROOT / "configs/research/grpo_contract_v1.json")


@pytest.fixture
def records():
    return [json.loads(line) for line in (ROOT / "data/research_v1_frozen/train.jsonl").read_text().splitlines()]


def choice(calls=None, content="wrong", tokens=(42, 43), finish="stop"):
    return Choice.model_validate({"index": 0, "finish_reason": finish,
        "message": {"role": "assistant", "content": content,
                    "tool_calls": None if calls is None else [{"id": f"c{i}", "type": "function", "function": {
                        "name": c["name"], "arguments": json.dumps(c["arguments"])}} for i, c in enumerate(calls)]},
        "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": -0.2,
                                  "bytes": None, "top_logprobs": []} for t in tokens]}})


def response(c):
    return SimpleNamespace(choices=[c], usage=SimpleNamespace(prompt_tokens=20, completion_tokens=len(c.logprobs.content)))


@pytest.mark.parametrize("key,value", [("seed", -1), ("seed", 42), ("group_size", 1), ("group_size", True),
    ("learning_rate", 0), ("beta", float("nan")), ("top_p", 1.1), ("temperature", 0),
    ("max_turns", 0), ("whole_group_retries", 2), ("formal_training_authorized", True),
    ("formal_training_authorized", 0), ("max_prompt_tokens", 4096)])
def test_reject_invalid_config(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_mapping_and_seed(config, records):
    c = effective_contract(config)
    assert c["train_kwargs"]["learning_rate"] == config["learning_rate"]
    assert c["train_kwargs"]["kl_penalty_coef"] == config["beta"]
    assert c["train_kwargs"]["allow_training_without_logprobs"] is False
    assert internal_art_config(config)["peft_args"]["random_state"] == config["seed"]
    assert internal_art_config(config)["engine_args"]["seed"] == config["seed"]
    assert scene_plan(records, config) == scene_plan(records, config)
    assert member_seed(config, 0, 0, 0) != member_seed(config, 0, 0, 1)
    assert len({r["sample_id"] for r in scene_plan(records, config)[0]}) == config["groups_per_step"]
    records[0]["split"] = "test"
    with pytest.raises(ValueError):
        scene_plan(records, config)


@pytest.mark.asyncio
async def test_multiturn_raw_choice_budget_and_usage(config, records):
    row = next(r for r in records if r["reference_depth"] == 3)
    choices = [choice([call]) for call in row["expected_calls"]] + [choice()]
    calls = []
    async def complete(**kwargs):
        calls.append(kwargs)
        return response(choices[len(calls) - 1])
    result = await training_rollout("arm@10", row, config, config["rewards"]["G2"], 99,
                                    complete=complete, count_prompt=lambda *_: 20)
    assert result.infra_error is None
    assert [x for x in result.messages_and_choices if not isinstance(x, dict)] == choices
    assert all(next(x for x in result.messages_and_choices if x is c) is c for c in choices)
    assert all(g["token_ids"] == [42, 43] and g["assistant_mask"] == [True, True] for g in result.generations)
    assert result.completion_tokens == 8 and result.prompt_tokens == 80 and result.requests == 4
    assert result.reward == 4
    assert [k["seed"] for k in calls] == [99, 100, 101, 102]
    assert all(k["logprobs"] and k["temperature"] == config["temperature"] and k["top_p"] == config["top_p"]
               and k["max_tokens"] == 512 and k["model"] == "arm@10" for k in calls)
    assert calls[1]["messages"][-1]["role"] == "tool"
    assert "environment_snapshot" not in json.dumps(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["infra", "empty_choices", "missing_logprobs", "usage", "token_id", "positive_logprob"])
async def test_infra_never_negative_reward(config, records, kind):
    async def complete(**kwargs):
        if kind == "infra":
            raise InfraFailure("network")
        c = choice()
        reply = response(c)
        if kind == "empty_choices":
            reply.choices = []
        elif kind == "missing_logprobs":
            c.logprobs = None
        elif kind == "usage":
            reply.usage = None
        elif kind == "token_id":
            assert c.logprobs is not None and c.logprobs.content is not None
            c.logprobs.content[0].token = c.logprobs.content[0].token.replace("token_id:", "text:")
        elif kind == "positive_logprob":
            assert c.logprobs is not None and c.logprobs.content is not None
            c.logprobs.content[0].logprob = 1
        return reply
    result = await training_rollout("arm", records[0], config, config["rewards"]["G1"], 0,
                                    complete=complete, count_prompt=lambda *_: 20)
    assert result.infra_error and result.reward is None
    assert result.requests == 1
    assert result.usage_unknown_requests == (kind in {"infra", "usage"})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["wrong_tool", "bad_json", "empty_text", "length", "turn_budget", "token_budget", "prompt_budget"])
async def test_model_errors_retained(config, records, kind):
    c = choice([{"name": "unknown_tool", "arguments": {}}]) if kind in {"wrong_tool", "bad_json", "turn_budget", "token_budget"} else choice(content="")
    if kind == "bad_json":
        assert c.message.tool_calls is not None
        first = c.message.tool_calls[0]
        assert first.type == "function"
        first.function.arguments = "{"
    if kind == "length":
        c.finish_reason = "length"
    if kind == "turn_budget":
        config["max_turns"] = 1
    if kind == "token_budget":
        config["max_total_completion_tokens"] = 2
    requests = []
    async def complete(**kwargs):
        requests.append(kwargs)
        return response(c)
    result = await training_rollout("arm", records[0], config, config["rewards"]["G2"], 0,
        complete=complete, count_prompt=lambda *_: 9999 if kind == "prompt_budget" else 20)
    assert result.infra_error is None and result.reward is not None
    if kind != "prompt_budget":
        assert result.messages_and_choices[2] is c
    if kind == "token_budget":
        assert requests[0]["max_tokens"] == 2 and len(requests) == 1


@pytest.mark.asyncio
async def test_programming_error_not_silently_infra(config, records):
    async def broken(**kwargs):
        raise ValueError("bug")
    with pytest.raises(ValueError, match="bug"):
        await training_rollout("arm", records[0], config, config["rewards"]["G1"], 0,
                               complete=broken, count_prompt=lambda *_: 20)


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [False, True])
async def test_whole_group_retry_no_member_filter(config, records, persist):
    count, seeds, events = 0, [], []
    async def rollout(row, seed):
        nonlocal count
        seeds.append(seed)
        i = count
        count += 1
        failed = i == 1 or (persist and i == config["group_size"] + 1)
        return TrainingRollout(reward=None if failed else -3.0, infra_error="network" if failed else None)
    result = await collect_group(records[0], config, 0, 0, rollout=rollout, log=events.append)
    assert count == 2 * config["group_size"]
    assert seeds[:config["group_size"]] == seeds[config["group_size"]:]
    assert sum(e["kind"] == "rollout" for e in events) == count
    if persist:
        assert result is None and events[-1]["skipped"]
    else:
        assert result is not None
        assert len(result) == config["group_size"] and all(m.reward == -3 for m in result)


@pytest.mark.asyncio
async def test_zero_variance_not_update(config, records):
    async def rollout(*_):
        return TrainingRollout(reward=-3.0)
    async def update(*_, **__):
        pytest.fail("must not increment fake ART checkpoint on zero signal")
    events = []
    totals = await run_steps(records, config, arm="G1", rollout=rollout, update=update, log=events.append)
    assert totals["effective_updates"] == totals["backend_calls"] == 0
    assert totals["zero_variance_groups"] == 4


@pytest.mark.asyncio
async def test_missing_update_evidence_rejected(config, records):
    i = 0
    async def rollout(*_):
        nonlocal i
        i += 1
        return TrainingRollout(reward=float(i % 2))
    async def update(*_, **__):
        return {"checkpoint_step": 11}
    with pytest.raises(ValueError, match="optimizer evidence"):
        await run_steps(records, config, arm="G1", rollout=rollout, update=update, log=lambda _: None)


@pytest.mark.parametrize("recipe", ["age_calculation", "weather_offset_convert", "country_density_convert", "search_and_convert", "population_math"])
def test_research_reward_dynamic_routes(records, recipe):
    # Composition-only validation recipes are synthetic checker fixtures here,
    # not production training data or any model generation/validation tuning.
    fixture_rows = records + [json.loads(line) for line in (ROOT / "data/research_v1_frozen/validation.jsonl").read_text().splitlines()]
    row = copy.deepcopy(next(r for r in fixture_rows if r["recipe_id"] == recipe))
    row["split"] = "train"
    good = copy.deepcopy(row["reference_trace"])
    b = score(row, good, "reward-v2-research-v1")
    assert b["total"] == 4 and b["dynamic"] > 0
    assert score(row, good, "reward-v1-legacy")["total"] < 4
    bad = copy.deepcopy(good)
    for s in bad:
        s["turn"] = 0
    b = score(row, bad, "reward-v2-research-v1")
    assert b["dynamic"] == 0 and b["total"] <= 2
    bad = copy.deepcopy(good)
    bad[0]["observation"] = "fabricated"
    assert score(row, bad, "reward-v2-research-v1")["total"] <= 2
    row["split"] = "test"
    with pytest.raises(ValueError):
        score(row, good, "reward-v2-research-v1")


def test_audit_cost_includes_discarded(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path)
    m = TrainingRollout(requests=2, prompt_tokens=40, completion_tokens=4, usage_unknown_requests=1, infra_error="network")
    audit({"kind": "rollout", "payload": m.audit()})
    audit({"kind": "group", "retry": True, "skipped": False})
    audit({"kind": "summary"})
    audit.close()
    summary = json.loads(path.read_text().splitlines()[-1])
    assert summary["cost"]["known_total_tokens"] == 44
    assert summary["cost"]["retries"] == 1
    assert not summary["cost"]["token_cost_complete"]
    assert summary["cost"]["GPU_allocation_hours"] is None
    with pytest.raises(FileExistsError):
        AuditLog(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("learning_rate", [5e-6, 5e-5])
async def test_real_cpu_optimizer_isolation_and_effective_lr(config, learning_rate):
    config["learning_rate"] = learning_rate
    torch = pytest.importorskip("torch")
    def fresh():
        with torch.random.fork_rng():
            torch.manual_seed(config["seed"])
            model = torch.nn.Linear(2, 2, bias=False)
        opt = torch.optim.AdamW(model.parameters(), lr=0.123, weight_decay=0)
        verify_fresh_optimizer(opt, list(model.parameters()))
        return model, opt
    a, oa = fresh()
    b, ob = fresh()
    assert torch.equal(a.weight, b.weight)
    verify_arm_isolation(a, oa, b, ob)
    with pytest.raises(ValueError):
        verify_arm_isolation(a, oa, a, oa)
    class FakeART:
        Trajectory = SimpleNamespace
        @staticmethod
        def TrajectoryGroup(trajectories):
            return SimpleNamespace(trajectories=trajectories)
    class Model:
        step = 10
        async def get_step(self):
            return self.step
    class CPUBackend:
        async def train(self, model, groups, **kw):
            for g in oa.param_groups:
                g["lr"] = kw["learning_rate"]
            rewards = torch.tensor([m.reward for m in groups[0].trajectories], dtype=torch.float32)
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            x = torch.tensor([[1., 0.], [0., 1.]])
            logp = a(x).log_softmax(-1)[:, 0]
            # Tiny CPU objective; actual ART beta behavior is checked by the
            # separate pinned-wheel loss contract, not this substitute backend.
            loss = -(advantages * logp).mean()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(a.parameters(), 1))
            oa.step()
            oa.zero_grad()
            model.step += 1
            return SimpleNamespace(step=model.step, checkpoint_path="cpu-fixture", metrics={"loss/train": float(loss.detach()), "loss/grad_norm": norm})
    update = MeasuredARTUpdate(CPUBackend(), Model(), oa, list(a.parameters()), FakeART, version="0.5.17")
    receipt = await update([[TrainingRollout(reward=-3), TrainingRollout(reward=4)]], **art_train_kwargs(config))
    assert receipt["optimizer_steps"] == 1 and receipt["changed_tensors"] == 1 and receipt["grad_norm"] > 0
    assert receipt["learning_rate"] == config["learning_rate"]
    assert not torch.equal(a.weight, b.weight) and not ob.state
    with pytest.raises(ValueError):
        verify_fresh_optimizer(oa, list(a.parameters()))


@pytest.mark.parametrize("corruption", [None, "source_changed", "wrong_shared_identity", "reuse"])
def test_arm_staging_isolated_and_immutable(tmp_path, config, corruption):
    from scripts.verify_research_eval_cpu import create_fixture_spec
    from src.research_model_identity import artifact_inventory, resolve_identity
    from src.research_sft_run import stage_grpo_starts
    spec = create_fixture_spec(tmp_path / "source", checkpoint=True)
    original = resolve_identity(spec)
    starts = stage_grpo_starts(spec, tmp_path / "starts", selection={"step": 7})
    if corruption == "source_changed":
        (Path(starts["arms"]["G1"]["model_spec"]["checkpoint_path"]) / "adapter_model.safetensors").write_bytes(b"changed")
    if corruption == "wrong_shared_identity":
        starts["source"]["fingerprint"] = "0" * 64
        (tmp_path / "starts/starts.json").write_text(json.dumps(starts))
    kwargs = {"code_revision": "a" * 40, "data_sha256": "b" * 64}
    if corruption in {"source_changed", "wrong_shared_identity"}:
        with pytest.raises(ValueError):
            prepare_arms(tmp_path / "starts/starts.json", tmp_path / "run", config, **kwargs)
        return
    receipt = prepare_arms(tmp_path / "starts/starts.json", tmp_path / "run", config, **kwargs)
    a, b = (receipt["arms"][arm] for arm in ("G1", "G2"))
    assert a["model_name"] != b["model_name"] and a["project"] != b["project"]
    assert a["art_path"] != b["art_path"] and a["workspace"] != b["workspace"]
    assert a["reward_version"] != b["reward_version"]
    assert a["identity"]["fingerprint"] == b["identity"]["fingerprint"] == original["fingerprint"]
    for arm in (a, b):
        assert not arm["optimizer_initialized"]
        assert not list(Path(arm["art_path"]).iterdir())
        assert all(not p.is_symlink() and p.stat().st_nlink == 1 for p in Path(arm["model_spec"]["checkpoint_path"]).iterdir())
    if corruption == "reuse":
        with pytest.raises(FileExistsError):
            prepare_arms(tmp_path / "starts/starts.json", tmp_path / "run", config, **kwargs)
    (Path(a["model_spec"]["checkpoint_path"]) / "adapter_model.safetensors").write_bytes(b"G1 training mutation")
    assert artifact_inventory(Path(b["model_spec"]["checkpoint_path"])) == original["checkpoint_files"]
    assert resolve_identity(spec) == original


@pytest.mark.asyncio
async def test_budget_timeout_partial_cost_and_sampling_override(config, records):
    import asyncio
    config.update(temperature=0.7, top_p=0.8, max_completion_tokens=4, max_turns=2, timeout_seconds=0.01)
    requests = []
    async def complete(**kwargs):
        requests.append(kwargs)
        if len(requests) == 2:
            await asyncio.sleep(1)
        return response(choice([{"name": "unknown", "arguments": {}}]))
    result = await training_rollout("arm@10", records[0], config, config["rewards"]["G2"], 0,
                                    complete=complete, count_prompt=lambda *_: 20)
    assert result.infra_error is not None and result.reward is None
    assert result.requests == 2 and result.completion_tokens == 2 and result.prompt_tokens == 20
    assert result.usage_unknown_requests == 1
    assert requests[0]["temperature"] == 0.7 and requests[0]["top_p"] == 0.8 and requests[0]["max_tokens"] == 4


def test_reward_never_uses_final_answer(records):
    row = copy.deepcopy(records[0])
    before = [score(row, row["reference_trace"], v) for v in ("reward-v1-legacy", "reward-v2-research-v1")]
    row.update(oracle_answer={"value": "poison"}, reference_answer={"value": "poison"}, answer_spec={})
    assert before == [score(row, row["reference_trace"], v) for v in ("reward-v1-legacy", "reward-v2-research-v1")]


@pytest.mark.asyncio
async def test_art_step_advance_without_learning_not_effective(config, records):
    counter = 0
    async def rollout(*_):
        nonlocal counter
        counter += 1
        return TrainingRollout(reward=float(counter % 2))
    async def update(*_, **kwargs):
        return {"optimizer_steps": 0, "changed_tensors": 0, "loss": 0., "grad_norm": 0.,
                "learning_rate": kwargs["learning_rate"], "beta": kwargs["kl_penalty_coef"],
                "checkpoint_step": 11, "checkpoint_path": "fixture-copy-not-learning"}
    totals = await run_steps(records, config, arm="G1", rollout=rollout, update=update, log=lambda _: None)
    assert totals["backend_calls"] == 2 and totals["effective_updates"] == totals["optimizer_steps"] == 0


def test_runtime_seed_affects_python_numpy_torch(config):
    import random
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    receipt = seed_runtime(config, torch=torch)
    first = [random.random(), float(numpy.random.rand()), float(torch.rand(1))]
    seed_runtime(config, torch=torch)
    assert first == [random.random(), float(numpy.random.rand()), float(torch.rand(1))]
    assert receipt["torch_CPU_seeded"] and receipt["python_seeded"] and receipt["numpy_seeded"]
    assert not receipt["torch_CUDA_seed_verified"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["G1", "G2"])
async def test_bound_rollout_reward_and_identity(config, records, arm):
    receipt = {"arm": arm, "model_name": f"isolated-{arm}", "project": f"project-{arm}", "reward_version": config["rewards"][arm]}
    model = SimpleNamespace(name=receipt["model_name"], project=receipt["project"], get_inference_name=lambda: receipt["model_name"] + "@10")
    tokenizer = SimpleNamespace(apply_chat_template=lambda *_, **__: list(range(20)))
    async def complete(**kwargs):
        return response(choice())
    bound = ConfiguredRollout(receipt, model, tokenizer, config, complete=complete)
    result = await bound(records[0], 4)
    assert result.reward_breakdown["version"] == config["rewards"][arm]
    assert result.reward == -3
    model.get_inference_name = lambda: "OTHER-ARM@10"
    with pytest.raises(ValueError, match="served model"):
        await bound(records[0], 4)
    model.project = "wrong-project"
    with pytest.raises(ValueError, match="identity"):
        ConfiguredRollout(receipt, model, tokenizer, config, complete=complete)


def test_art_view_preserves_frozen_schema_order_and_rejects_unknown():
    from src.research_environment import TOOL_SCHEMAS
    seen = []
    raw = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: seen.append(kwargs) or [1])
    view = ARTTokenizerView(raw)
    reordered = [{"function": t["function"], "type": t["type"]} for t in TOOL_SCHEMAS]
    wrapped = [{"type": "function", "function": t} for t in reordered]
    view.apply_chat_template([], tools=wrapped)
    assert list(seen[0]["tools"][0]) == list(TOOL_SCHEMAS[0]) == ["type", "function"]
    with pytest.raises(ValueError, match="frozen research schemas"):
        view.apply_chat_template([], tools=[{"type": "function", "function": {"name": "other"}}])

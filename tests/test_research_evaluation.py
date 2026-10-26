"""R2.5–R2.8 CPU contracts: fake weights/transports, no torch/GPU/model downloads."""

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval_research as entry
from scripts import verify_research_eval_cpu as fixtures
from src.research_eval_backend import ModelProtocolError, decode_qwen
from src.research_eval_metrics import (
    cluster_interval,
    paired_difference,
    summarize_evaluation,
    wilson,
)
from src.research_eval_protocol import (
    load_development_records,
    plan_attempts,
    protocol_identity,
    request_seed,
    validate_protocol,
)
from src.research_evaluation import complete_ledger, evaluate_records
from src.research_model_identity import (
    file_hash,
    resolve_identity,
    verify_receipt,
    verify_unchanged,
)
from src.research_schema import load_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return load_json(ROOT / "configs/research/evaluation_v1.json")


@pytest.fixture(scope="module")
def development():
    return load_development_records(ROOT / "data/research_v1_candidate")


@pytest.fixture
def identity(tmp_path):
    return resolve_identity(fixtures.create_fixture_spec(tmp_path))


def backend_for(identity, rows, protocol, *, fault=None):
    plan = plan_attempts(rows, protocol)
    responses = {p["attempt_id"]: fixtures.scripted_responses(rows[p["scenario_index"]], fault=fault) for p in plan}
    return fixtures.FixtureBackend(identity, responses)


@pytest.mark.parametrize("key,value", [
    ("protocol_version", "future"), ("sample_k", 7), ("sample_k", True), ("seed", -1),
    ("modes", []), ("modes", ["sample", "deterministic"]), ("max_turns", 7), ("max_calls_per_turn", 9),
    ("max_new_tokens_per_turn", 0), ("max_output_tokens_per_attempt", True), ("max_prompt_tokens", 0),
    ("request_timeout_seconds", 0), ("request_timeout_seconds", float("nan")), ("concurrency", 0),
    ("concurrency", 17), ("chat_template_policy", "guess"), ("tool_codec", "repair-json"),
    ("infra_policy", "drop-and-retry"), ("test_policy", "allow-test"),
    ("deterministic", {"temperature": .1, "top_p": 1}), ("deterministic", {"temperature": 0, "top_p": .9}),
    ("sample", {"temperature": 0, "top_p": .95}), ("sample", {"temperature": 1, "top_p": 0}),
    ("sample", {"temperature": 1, "top_p": 1, "ignored": 1}),
])
def test_invalid_protocol_fails_closed(protocol, key, value):
    protocol[key] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_protocol_seeds_full_schedule_and_no_inferred_clamping(protocol, development):
    rows, _ = development
    plan = plan_attempts(rows, protocol)
    assert len(plan) == 180 * 9 and len({p["attempt_id"] for p in plan}) == len(plan)
    assert len({request_seed(protocol, p, 0) for p in plan}) == len(plan)
    assert request_seed(protocol, plan[0], 0) != request_seed(protocol, plan[0], 1)
    assert plan == plan_attempts(copy.deepcopy(rows), copy.deepcopy(protocol))
    altered = copy.deepcopy(protocol)
    altered["max_prompt_tokens"] += 1
    assert protocol_identity(altered)["sha256"] != protocol_identity(protocol)["sha256"]
    for bad in ([], [rows[0], rows[0]], [{**rows[0], "split": "test"}]):
        with pytest.raises(ValueError):
            plan_attempts(bad, protocol)
    for split, limit in (("test", None), ("validation", 0), ("validation", True), ("validation", 181)):
        with pytest.raises(ValueError):
            load_development_records(ROOT / "data/research_v1_candidate", split=split, limit=limit)


def test_data_manifest_bytes_counts_and_no_label_messages(tmp_path, development):
    row = development[0][0]
    directory = tmp_path / "data"
    directory.mkdir()
    content = json.dumps(row) + "\n"
    (directory / "validation.jsonl").write_text(content)
    manifest = {"dataset_version": "fixture", "counts": {"validation": 1},
                "files": {"validation.jsonl": {"sha256": file_hash(directory / "validation.jsonl"), "bytes": len(content.encode())}}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert len(load_development_records(directory)[0]) == 1
    (directory / "validation.jsonl").write_text(content + "\n")
    with pytest.raises(ValueError, match="bytes"):
        load_development_records(directory)


@pytest.mark.parametrize("mutation", ["main_revision", "base_with_checkpoint", "adapter_in_base", "empty_weight", "pickle", "remote_code", "origin",
                                     "model_type", "missing_tokenizer", "missing_template", "revision_conflict", "extra_tensor"])
def test_base_identity_rejects_wrong_or_ambiguous_assets(tmp_path, mutation):
    spec = fixtures.create_fixture_spec(tmp_path)
    base = Path(spec["base_path"])
    if mutation == "main_revision":
        spec["base_revision"] = "main"
    elif mutation == "base_with_checkpoint":
        spec["checkpoint_path"] = str(base)
    elif mutation == "adapter_in_base":
        (base / "adapter_config.json").write_text("{}")
    elif mutation == "empty_weight":
        (base / "model.safetensors").write_bytes(b"")
    elif mutation == "pickle":
        (base / "pytorch_model.bin").write_bytes(b"unsafe")
    elif mutation == "remote_code":
        (base / "model.py").write_text("raise AssertionError")
    elif mutation == "origin":
        (base / "research_model_origin.json").write_text(json.dumps({"model_id": "Wrong", "revision": spec["base_revision"]}))
    elif mutation in {"model_type", "revision_conflict"}:
        config = load_json(base / "config.json")
        config.update({"model_type": "wrong"} if mutation == "model_type" else {"_commit_hash": "b" * 40})
        (base / "config.json").write_text(json.dumps(config))
    elif mutation == "missing_tokenizer":
        (base / "tokenizer.json").unlink()
    elif mutation == "missing_template":
        (base / "tokenizer_config.json").write_text("{}")
    else:
        (base / "unrelated.safetensors").write_bytes(b"extra")
    with pytest.raises(ValueError):
        resolve_identity(spec)


@pytest.mark.parametrize("mutation", ["step", "bool_step", "base_binding", "state_step", "adapter_type", "adapter_base", "revision", "tokenizer", "directory_step"])
def test_checkpoint_explicit_step_base_and_tokenizer_binding(tmp_path, mutation):
    spec = fixtures.create_fixture_spec(tmp_path, checkpoint=True)
    checkpoint = Path(spec["checkpoint_path"])
    if mutation in {"step", "bool_step"}:
        spec["checkpoint_step"] = True if mutation == "bool_step" else 8
    elif mutation == "base_binding":
        binding = load_json(checkpoint / "research_checkpoint.json")
        binding["base_fingerprint"] = "0" * 64
        (checkpoint / "research_checkpoint.json").write_text(json.dumps(binding))
    elif mutation == "state_step":
        (checkpoint / "trainer_state.json").write_text('{"global_step":99}')
    elif mutation in {"adapter_type", "adapter_base", "revision"}:
        config = load_json(checkpoint / "adapter_config.json")
        config.update({"peft_type": "OTHER"} if mutation == "adapter_type" else {"base_model_name_or_path": "Wrong"} if mutation == "adapter_base" else {"revision": "b" * 40})
        (checkpoint / "adapter_config.json").write_text(json.dumps(config))
    elif mutation == "tokenizer":
        (checkpoint / "tokenizer_config.json").write_text('{"chat_template":"different"}')
    else:
        new_path = checkpoint.with_name("checkpoint-9")
        checkpoint.rename(new_path)
        spec["checkpoint_path"] = str(new_path)
    with pytest.raises(ValueError):
        resolve_identity(spec)


def test_weight_shard_and_symlink_safety(tmp_path):
    spec = fixtures.create_fixture_spec(tmp_path / "good")
    base = Path(spec["base_path"])
    (base / "model.safetensors").unlink()
    (base / "model-00001.safetensors").write_bytes(fixtures.FIXTURE_MARKER)
    index = {"weight_map": {"layer": "model-00001.safetensors"}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    assert resolve_identity(spec)["base_files"]["model-00001.safetensors"]
    index["weight_map"]["layer"] = "../secret.safetensors"
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        resolve_identity(spec)
    spec = fixtures.create_fixture_spec(tmp_path / "link")
    base = Path(spec["base_path"])
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (base / "tokenizer.json").unlink()
    (base / "tokenizer.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        resolve_identity(spec)


@pytest.mark.parametrize("field,value", [("base_path", "/wrong"), ("checkpoint_path", "/wrong"), ("base_fingerprint", "wrong"),
                                         ("checkpoint_step", 8), ("active_adapters", ["old_lora"]), ("device", "cuda:0"),
                                         ("dtype", "float16"), ("chat_template_sha256", "b" * 64), ("real_model_loaded", True)])
def test_actual_loader_receipt_mismatch_rejected(identity, field, value):
    receipt = fixtures.fixture_receipt(identity)
    receipt[field] = value
    with pytest.raises(ValueError):
        verify_receipt(identity, receipt)


def test_no_silent_identity_mutation(tmp_path):
    spec = fixtures.create_fixture_spec(tmp_path)
    identity = resolve_identity(spec)
    verify_unchanged(spec, identity)
    (Path(spec["base_path"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_unchanged(spec, identity)


@pytest.mark.parametrize("text", ["<tool_call>{}</tool_call>", "<tool_call>{\"name\":\"calculator\",\"arguments\":{},\"extra\":1}</tool_call>",
                                  "<tool_call>{\"name\":\"calculator\",\"arguments\":{\"x\":NaN}}</tool_call>",
                                  "<tool_call>{\"name\":\"x\",\"name\":\"y\",\"arguments\":{}}</tool_call>",
                                  "<tool_call>{\"name\":\"calculator\",\"arguments\":\"{}\"}</tool_call>",
                                  "<tool_call>{\"name\":\"calculator\",\"arguments\":{}}", "</tool_call>", "<tool_call><tool_call>{}</tool_call>"])
def test_qwen_codec_no_repair(text):
    with pytest.raises(ModelProtocolError):
        decode_qwen(text, 0)


def test_qwen_codec_raw_final_and_generated_id_provenance():
    text = 'Thinking. <tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>'
    message = decode_qwen(text, 2)
    assert message["content"] == "Thinking."
    assert message["tool_calls"][0]["id"] == "qwen-turn-2-call-0"
    assert message["tool_calls"][0]["function"]["arguments"] == {"expression": "2+2"}
    assert decode_qwen('```json\n{"value":4,"unit":"dimensionless"}\n```', 0)["content"].startswith("```")


@pytest.mark.parametrize("fault,answer_ok,infra,valid,strict", [
    (None, True, False, True, True), ("wrong_unit", False, False, True, True),
    ("wrong_tool", True, False, True, False), ("same_turn", True, False, True, False),
    ("length", False, False, True, False), ("invalid_arguments", False, False, True, False),
    ("direct_or_alternative", True, False, True, False), ("empty", False, True, False, False),
])
def test_termination_model_errors_never_filtered_and_answer_route_independent(protocol, development, identity, fault, answer_ok, infra, valid, strict):
    row = next(r for r in development[0] if r["recipe_id"] == "country_density_convert")
    protocol["modes"] = ["deterministic"]
    backend = backend_for(identity, [row], protocol, fault=fault)
    plan, results = asyncio.run(evaluate_records([row], protocol, identity, backend))
    result = results[0]
    assert result["checker"]["answer"]["correct"] == answer_ok
    assert result["infra_failure"] == infra and result["valid_attempt"] == valid
    assert result["checker"]["trace"]["strict_complete"] == strict
    assert result["attempt_id"] == plan[0]["attempt_id"] and result["received"]
    assert backend.closed
    for event in result["tool_events"]:
        assert event["observation_source"] in {"local-research-executor", "argument-parser-rejection"}
        assert event["attempt_id"] == result["attempt_id"]
    if fault == "same_turn":
        assert {e["turn"] for e in result["tool_events"]} == {0}
        assert "same_turn_dependency" in result["errors"]
    for request in backend.seen:
        assert set(request["messages"][0]) == {"role", "content"}
        assert "oracle_answer" not in json.dumps(request) and "reference_trace" not in json.dumps(request)
        assert request["messages"][1] == {"role": "user", "content": row["task"]}


def test_incomplete_returns_pass8_not_rollout_average_and_cluster_ci(protocol, development, identity):
    rows = development[0][:2]
    backend = backend_for(identity, rows, protocol)
    for slot in plan_attempts(rows, protocol):
        if slot["mode"] == "sample" and (slot["scenario_index"] == 1 or slot["index"] != 0):
            backend.responses[slot["attempt_id"]] = fixtures.scripted_responses(rows[slot["scenario_index"]], fault="wrong_unit")
    plan, outcomes = asyncio.run(evaluate_records(rows, protocol, identity, backend))
    ledger = complete_ledger(rows, plan, outcomes)
    summary = summarize_evaluation(ledger, protocol)
    sample = summary["evaluations"]["sample"]["overall"]
    assert sample["observed_answer_pass_at_8"]["rate"] == .5
    assert sample["answer_rollout_accuracy_all"]["rate"] == 1 / 16
    assert sample["observed_answer_pass_at_8"]["ci95"] == pytest.approx(wilson(1, 2))
    assert sample["answer_rollout_accuracy_all"]["ci_method"].startswith("semantic-scenario")
    missing = complete_ledger(rows, plan, outcomes[:-1])
    summary = summarize_evaluation(missing, protocol)
    assert not summary["comparison_valid"]
    sample = summary["evaluations"]["sample"]["overall"]
    assert sample["planned_attempts"] == 16 and sample["missing_returned_attempts"] == 1
    assert sample["observed_answer_pass_at_8"]["denominator"] == 2
    assert sum(sample["primary_errors"].values()) == 16
    for wrong in (outcomes + outcomes[:1], [{**outcomes[0], "attempt_id": "unknown"}], [{**outcomes[0], "family": "substituted"}]):
        with pytest.raises(ValueError):
            complete_ledger(rows, plan, wrong)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_evaluation(outcomes[:-1], protocol)
    assert paired_difference(ledger, ledger)["second_minus_first"] == 0
    assert paired_difference(ledger, ledger, mode="sample")["ci95"] == [0, 0]
    assert wilson(1, 2) == pytest.approx([.0945312057, .9054687943])
    assert wilson(0, 0) is None and cluster_interval([]) is None


def test_fixed_sampling_seeds_concurrency_and_budget(protocol, development, identity):
    rows = development[0][:3]
    protocol["concurrency"] = 2
    backend = backend_for(identity, rows, protocol)
    backend.delay = .001
    plan, outcomes = asyncio.run(evaluate_records(rows, protocol, identity, backend))
    assert len(outcomes) == 27 and backend.peak == 2
    assert [r["attempt_id"] for r in outcomes] == [p["attempt_id"] for p in plan]
    for request in backend.seen:
        slot = next(p for p in plan if p["attempt_id"] == request["attempt_id"])
        assert request["temperature"] == protocol[slot["mode"]]["temperature"]
        assert request["top_p"] == protocol[slot["mode"]]["top_p"]
        assert request["seed"] == request_seed(protocol, slot, request["turn"])
        assert request["max_new_tokens"] <= protocol["max_new_tokens_per_turn"]
    protocol["modes"] = ["deterministic"]
    row = next(r for r in development[0] if r["recipe_id"] == "country_density_convert")
    for field in ("max_turns", "max_output_tokens_per_attempt"):
        p = copy.deepcopy(protocol)
        p[field] = 1 if field == "max_turns" else 8
        backend = backend_for(identity, [row], p)
        _, out = asyncio.run(evaluate_records([row], p, identity, backend))
        assert out[0]["termination"] == "truncated" and not out[0]["checker"]["answer"]["correct"]
        assert len(backend.seen) == 1


def test_request_timeout_no_retry_and_shutdown(protocol, development, identity):
    protocol["modes"] = ["deterministic"]
    protocol["request_timeout_seconds"] = .001
    row = development[0][0]
    backend = backend_for(identity, [row], protocol)
    backend.delay = .03
    _, outcomes = asyncio.run(evaluate_records([row], protocol, identity, backend))
    assert len(backend.seen) == 1 and backend.closed and backend.active == 0
    assert outcomes[0]["infra_failure"] and outcomes[0]["requests"][0]["error"] == "request_timeout"


def test_loader_default_preflight_run_archive_replay_and_tamper(tmp_path, protocol, development):
    rows, dataset = development
    rows = rows[:1]
    dataset = {**dataset, "selected_count": 1, "selected_sample_ids": [rows[0]["sample_id"]], "selection": "explicit-first-1-development-subset"}
    spec = fixtures.create_fixture_spec(tmp_path / "models", checkpoint=True)
    output = tmp_path / "preflight"

    def forbidden(*args):
        raise AssertionError("preflight must not load model")

    manifest = asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output, loader=forbidden))
    assert manifest["state"] == "preflight-only" and manifest["loader_receipt"] is None
    assert len(json.loads((output / "plan.json").read_text())) == 9
    with pytest.raises(ValueError, match="exists"):
        asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output))
    with pytest.raises(ValueError):
        asyncio.run(entry.audit_saved_run(output))
    output = tmp_path / "full-run"

    def loader(s, identity, p):
        return backend_for(identity, rows, p)

    manifest = asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output, run_cpu=True, loader=loader))
    summary = asyncio.run(entry.audit_saved_run(output))
    assert summary["comparison_valid"] and not manifest["loader_receipt"]["real_model_loaded"]
    raw = entry.read_jsonl(output / "attempts.jsonl")
    assert len(raw) == 9 and raw[0]["requests"][0]["raw_response"]
    raw[0]["checker"]["answer"]["correct"] = False
    (output / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    with pytest.raises(ValueError, match="bytes"):
        asyncio.run(entry.audit_saved_run(output))
    # Even updating the unauthenticated checksum cannot turn a forged result into replay evidence.
    manifest["artifacts"]["attempts.jsonl"] = {"sha256": file_hash(output / "attempts.jsonl"), "bytes": (output / "attempts.jsonl").stat().st_size}
    entry.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="replay"):
        asyncio.run(entry.audit_saved_run(output))


def test_loader_failure_and_cancellation_preserve_plan_missing_denominator(tmp_path, protocol, development):
    rows, dataset = development
    rows = rows[:1]
    spec = fixtures.create_fixture_spec(tmp_path / "models")

    def failed(*args):
        raise RuntimeError("deliberate load failure")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output, run_cpu=True, loader=failed))
    assert load_json(output / "manifest.json")["state"] == "failed"
    summary = load_json(output / "summary.json")
    assert not summary["comparison_valid"]
    assert summary["evaluations"]["sample"]["overall"]["planned_attempts"] == 8
    assert summary["evaluations"]["sample"]["overall"]["missing_returned_attempts"] == 8
    assert len(entry.read_jsonl(output / "ledger.jsonl")) == 9


def test_cpu_artifacts_reproducible_no_model_claims():
    files = fixtures.build_artifacts()
    assert files == fixtures.build_artifacts()
    report = json.loads(files["CPU_VERIFICATION.json"])
    assert not report["GPU_started"] and not report["test_used"]
    for kind in ("base", "checkpoint"):
        assert report["runs"][kind] == {"planned": 72, "returned": 72, "valid": 70, "infra": 2,
                                       "offline_replay_passed": True, "real_model_loaded": False}
        manifest = json.loads(files[f"{kind}/manifest.json"])
        assert not manifest["loader_receipt"]["real_model_loaded"]
        assert manifest["auto_retry_count"] == 0
    assert "不是" in files["REPORT.md"].decode()


def test_cli_no_import_heavy_defaults_and_forbidden_outputs(tmp_path):
    for command in ([sys.executable, "scripts/eval_research.py", "--help"],
                    [sys.executable, "-c", "import scripts.eval_research,sys; assert not any(k in sys.modules for k in ['torch','transformers','peft','art']);print('clean')"]):
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr
    for output in (str(tmp_path), "data/research_v1_candidate", "results/research"):
        with pytest.raises(ValueError):
            entry.checked_output(output)
    p = subprocess.run([sys.executable, "scripts/eval_research.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert p.returncode == 1 and "explicit local" in p.stderr


@pytest.mark.parametrize("content", ["", None])
def test_returned_empty_model_choice_is_model_failure_not_filtered(protocol, development, identity, content):
    protocol["modes"] = ["deterministic"]
    rows = development[0][:1]
    backend = backend_for(identity, rows, protocol)
    slot = plan_attempts(rows, protocol)[0]
    backend.responses[slot["attempt_id"]] = [{"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                                               "usage": {"prompt_tokens": 10, "completion_tokens": 1}}]
    _, outcomes = asyncio.run(evaluate_records(rows, protocol, identity, backend))
    result = outcomes[0]
    assert result["received"] and result["valid_attempt"] and not result["infra_failure"]
    assert not result["checker"]["answer"]["correct"] and "empty_model_output" in result["errors"]
    summary = summarize_evaluation(outcomes, protocol)
    assert summary["comparison_valid"]
    assert summary["evaluations"]["deterministic"]["overall"]["valid_attempts"] == 1


def test_metrics_reject_entire_missing_mode_or_empty_run(protocol, development, identity):
    rows = development[0][:1]
    _, outcomes = asyncio.run(evaluate_records(rows, protocol, identity, backend_for(identity, rows, protocol)))
    for ledger in ([], [r for r in outcomes if r["mode"] == "sample"], [r for r in outcomes if r["mode"] == "deterministic"],
                   outcomes + [{**outcomes[0], "attempt_id": "new", "mode": "unknown"}]):
        with pytest.raises(ValueError, match="incomplete"):
            summarize_evaluation(ledger, protocol)


def test_implicit_quantized_config_cannot_initialize_gpu_loader(tmp_path):
    spec = fixtures.create_fixture_spec(tmp_path)
    path = Path(spec["base_path"]) / "config.json"
    config = load_json(path)
    config["quantization_config"] = {"load_in_4bit": True}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quantized/GPU"):
        resolve_identity(spec)


def test_backend_close_failure_still_saves_full_failure_ledger(tmp_path, protocol, development):
    rows, dataset = development
    rows = rows[:1]
    spec = fixtures.create_fixture_spec(tmp_path / "models")

    class CloseFailure(fixtures.FixtureBackend):
        async def close(self):
            self.closed = True
            raise RuntimeError("deliberate close failure")

    def loader(s, identity, p):
        return CloseFailure(identity, backend_for(identity, rows, p).responses)

    output = tmp_path / "close-failure"
    with pytest.raises(RuntimeError):
        asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output, run_cpu=True, loader=loader))
    manifest = load_json(output / "manifest.json")
    summary = load_json(output / "summary.json")
    assert manifest["state"] == "failed" and manifest["close_error_type"] == "RuntimeError"
    assert len(entry.read_jsonl(output / "attempts.jsonl")) == 9
    assert len(entry.read_jsonl(output / "ledger.jsonl")) == 9 and not summary["comparison_valid"]


def test_cancelled_run_saves_plan_and_missing_placeholders(tmp_path, protocol, development):
    rows, dataset = development
    rows = rows[:1]
    spec = fixtures.create_fixture_spec(tmp_path / "models")

    class Cancelled(fixtures.FixtureBackend):
        async def complete(self, request):
            raise asyncio.CancelledError()

    def loader(s, identity, p):
        return Cancelled(identity, {})

    output = tmp_path / "cancelled"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output, run_cpu=True, loader=loader))
    assert load_json(output / "manifest.json")["state"] == "interrupted"
    ledger = entry.read_jsonl(output / "ledger.jsonl")
    assert len(ledger) == 9 and all(not r["received"] for r in ledger)
    summary = load_json(output / "summary.json")
    assert not summary["comparison_valid"] and summary["evaluations"]["sample"]["overall"]["planned_attempts"] == 8


@pytest.mark.parametrize("tamper", ["checksum", "version", "selected_count"])
def test_audit_rejects_weakened_manifest(tmp_path, protocol, development, tamper):
    rows, dataset = development
    rows = rows[:1]
    dataset = {**dataset, "selected_count": 1, "selected_sample_ids": [rows[0]["sample_id"]]}
    spec = fixtures.create_fixture_spec(tmp_path / "models")

    def loader(s, identity, p):
        return backend_for(identity, rows, p)

    output = tmp_path / "audit"
    manifest = asyncio.run(entry.run_local_evaluation(spec, protocol, rows, dataset, output, run_cpu=True, loader=loader))
    if tamper == "checksum":
        del manifest["artifacts"]["plan.json"]
    elif tamper == "version":
        manifest["checker_versions"]["answer"] = "another-checker"
    else:
        manifest["dataset"]["selected_count"] = 180
    entry.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        asyncio.run(entry.audit_saved_run(output))

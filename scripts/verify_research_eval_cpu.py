#!/usr/bin/env python3
"""Generate reproducible CPU-only eval contracts using EXPLICITLY FAKE weights/responses."""

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_research as entry  # noqa: E402
from scripts.diagnose_research import scripted_attempt  # noqa: E402
from src.research_environment import canonical_json  # noqa: E402
from src.research_eval_protocol import load_development_records, plan_attempts  # noqa: E402
from src.research_model_identity import MODEL_ID, resolve_identity  # noqa: E402
from src.research_schema import digest, load_json  # noqa: E402

FIXTURE_REVISION = "a" * 40
FIXTURE_MARKER = b"CPU-CONTRACT-FIXTURE-NOT-VALID-TENSORS-NOT-A-MODEL"


def create_fixture_spec(root: Path, *, checkpoint: bool = False) -> dict:
    base = root / "base"
    base.mkdir(parents=True, exist_ok=True)
    artifacts = {"config.json": {"model_type": "qwen2", "hidden_size": 2, "num_hidden_layers": 1},
                 "tokenizer_config.json": {"chat_template": "CPU-CONTRACT-FIXTURE-NOT-A-REAL-QWEN-TEMPLATE"},
                 "tokenizer.json": {"fixture": True},
                 "research_model_origin.json": {"model_id": MODEL_ID, "revision": FIXTURE_REVISION}}
    for name, value in artifacts.items():
        (base / name).write_text(canonical_json(value) + "\n", encoding="utf-8")
    (base / "model.safetensors").write_bytes(FIXTURE_MARKER)
    spec = {"kind": "base", "base_path": str(base), "base_revision": FIXTURE_REVISION,
            "checkpoint_path": None, "checkpoint_step": None}
    if checkpoint:
        identity = resolve_identity(spec)
        adapter = root / "checkpoint-7"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(FIXTURE_MARKER + b"-ADAPTER")
        (adapter / "adapter_config.json").write_text(canonical_json({"peft_type": "LORA", "base_model_name_or_path": MODEL_ID}), encoding="utf-8")
        (adapter / "research_checkpoint.json").write_text(canonical_json({"step": 7, "base_fingerprint": identity["base_fingerprint"],
                                                                         "base_revision": FIXTURE_REVISION}), encoding="utf-8")
        spec.update(kind="checkpoint", checkpoint_path=str(adapter), checkpoint_step=7)
    return spec


def fixture_receipt(identity: dict) -> dict:
    return {"backend": "cpu-fixture", "real_model_loaded": False,
            **{key: identity[key] for key in ("base_path", "checkpoint_path", "base_fingerprint", "checkpoint_fingerprint",
                                              "checkpoint_step", "base_config_hash", "chat_template_sha256")},
            "active_adapters": [] if identity["kind"] == "base" else ["research_eval"],
            "model_class": "ScriptedCPUFixtureNotModel", "device": "cpu", "dtype": "float32", "versions": {"fixture": "v1"},
            "runtime_config_sha256": digest({"fixture": True})}


def completion(message: dict, *, finish: str = "stop", tokens: int = 8) -> dict:
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 32, "completion_tokens": tokens}, "fixture_response": True}


def scripted_responses(row: dict, *, fault: str | None = None) -> list[dict]:
    attempt = scripted_attempt(row)
    calls = [{"id": f"fixture-call-{i}", "type": "function", "function": {"name": step["name"], "arguments": copy.deepcopy(step["arguments"])}}
             for i, step in enumerate(attempt["trace"])]
    final = {"role": "assistant", "content": attempt["content"]}
    if fault == "empty":
        return [{"choices": [], "usage": {"prompt_tokens": 32, "completion_tokens": 0}}]
    if fault == "wrong_unit":
        try:
            answer = json.loads(final["content"])
        except ValueError as exc:
            raise ValueError("scripted reference returned invalid final JSON") from exc
        answer["unit"] = "years" if answer["unit"] != "years" else "km"
        final["content"] = canonical_json(answer)
    elif fault == "wrong_tool":
        calls[0]["function"]["name"] = "unknown_tool"
    elif fault == "same_turn":
        return [completion({"role": "assistant", "content": None, "tool_calls": calls}, finish="tool_calls"), completion(final)]
    elif fault == "invalid_arguments":
        calls[0]["function"]["arguments"] = '{"value":1,"value":2}'
    elif fault == "direct_or_alternative":
        if row["recipe_id"] != "weather_convert":
            return [completion(final)]
        calls = [{"id": "fixture-fahrenheit", "type": "function", "function": {"name": "get_weather",
                  "arguments": {"city": row["parameters"]["entity"], "units": "fahrenheit"}}}]
    responses = [completion({"role": "assistant", "content": None, "tool_calls": [call]}, finish="tool_calls") for call in calls]
    responses.append(completion(final, finish="length" if fault == "length" else "stop"))
    return responses


class FixtureBackend:
    """Precomputed scripted responses are test controls, NEVER actual model predictions."""

    def __init__(self, identity: dict, responses: dict[str, list[dict]], *, delay: float = 0):
        self.receipt = fixture_receipt(identity)
        self.responses = copy.deepcopy(responses)
        self.delay = delay
        self.closed, self.active, self.peak = False, 0, 0
        self.seen = []

    async def complete(self, request: dict) -> dict:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.seen.append(copy.deepcopy(request))
        try:
            await asyncio.sleep(self.delay)
            return copy.deepcopy(self.responses[request["attempt_id"]][request["turn"]])
        finally:
            self.active -= 1

    async def close(self) -> None:
        self.closed = True


def select_fixture_rows(rows: list[dict]) -> list[dict]:
    selectors = [("simple_math", "iid"), ("simple_weather", "parameter"), ("age_calculation", "wording"),
                 ("country_density_convert", "iid"), ("weather_convert", "parameter"), ("weather_compare", "iid"),
                 ("gdp_scale", "composition"), ("weather_offset_convert", "composition")]
    return [next(r for r in rows if r["recipe_id"] == recipe and r["generalization"] == kind) for recipe, kind in selectors]


async def build_artifacts_async() -> dict[str, bytes]:
    protocol = load_json(ROOT / "configs/research/evaluation_v1.json")
    all_rows, dataset = load_development_records(ROOT / "data/research_v1_candidate")
    rows = select_fixture_rows(all_rows)
    dataset.update(selected_count=len(rows), selection="explicit-8-scripted-CPU-contract-fixtures-NOT-benchmark",
                   selected_sample_ids=[r["sample_id"] for r in rows])
    plan = plan_attempts(rows, protocol)
    responses = {}
    for slot in plan:
        fault = None
        if slot["mode"] == "sample":
            fault = {1: "wrong_unit", 2: "wrong_tool", 3: "same_turn", 4: "length", 5: "invalid_arguments", 6: "direct_or_alternative"}.get(slot["index"])
            if (slot["scenario_index"], slot["index"]) in {(0, 0), (1, 7)}:
                fault = "empty"
        responses[slot["attempt_id"]] = scripted_responses(rows[slot["scenario_index"]], fault=fault)
    files, counts = {}, {}
    with tempfile.TemporaryDirectory(prefix="research-eval-CPU-contract-") as tmp:
        temporary = Path(tmp)
        for kind in ("base", "checkpoint"):
            spec = create_fixture_spec(temporary / "model-fixtures" / kind, checkpoint=kind == "checkpoint")
            output = temporary / "runs" / kind

            def loader(s, identity, p):
                return FixtureBackend(identity, responses)

            manifest = await entry.run_local_evaluation(spec, protocol, rows, dataset, output, run_cpu=True, loader=loader)
            summary = await entry.audit_saved_run(output)
            if manifest["loader_receipt"]["real_model_loaded"]:
                raise ValueError("fixture must not claim actual model loading")
            det, sample = (summary["evaluations"][mode]["overall"] for mode in ("deterministic", "sample"))
            if det["planned_attempts"] != 8 or sample["planned_attempts"] != 64 or sample["infra_failures"] != 2 or summary["comparison_valid"]:
                raise ValueError("CPU contract denominators/failure policy drift")
            counts[kind] = {"planned": 72, "returned": det["received_attempts"] + sample["received_attempts"],
                            "valid": det["valid_attempts"] + sample["valid_attempts"], "infra": sample["infra_failures"],
                            "offline_replay_passed": True, "real_model_loaded": False}
            for path in sorted(output.iterdir()):
                content = path.read_text(encoding="utf-8").replace(str(temporary), "<EPHEMERAL-CPU-FIXTURE-NOT-MODEL>")
                files[f"{kind}/{path.name}"] = content.encode()
    report = {"kind": "CPU-mock-contract-verification-not-model-evaluation", "GPU_started": False, "test_used": False,
              "selected_validation_scenarios": 8, "strata": ["iid", "parameter", "wording", "composition"],
              "reference_depths": [1, 2, 3], "runs": counts,
              "controls": ["truthful executed route", "wrong unit but full protocol", "wrong tool but correct answer", "same-turn dependent calls",
                           "generation length", "duplicate argument keys", "direct correct answer/legal Fahrenheit bypass", "empty choices infra"],
              "limitations": ["Fake safetensors are explicit temporary placeholders, never loaded or published as weights.",
                              "Fixture receipts attest NO real model; paths redacted because temporary fixtures are removed.",
                              "--check regenerates these fixture artifacts; production --audit tested before fixture removal.",
                              "Real Qwen/PEFT/ART/vLLM loading and GPU decoding remain unverified."]}
    files["CPU_VERIFICATION.json"] = (canonical_json(report) + "\n").encode()
    files["REPORT.md"] = ("# R2.5–R2.8 本地评测 CPU 合同验收\n\n"
                          "**这是模拟响应合同测试，不是 Base/checkpoint 模型准确率，不是训练增益。**\n\n"
                          "- 同一入口分别处理显式 Base 和 step7 checkpoint 身份，文件字节/加载回执不混用。\n"
                          "- 8个validation开发案例覆盖1/2/3-call与4泛化分层；每个身份72个计划attempt（8 deterministic +64 sample）。\n"
                          "- 每个身份72/72返回、70/72有效、2个infra；全部保留分母、comparison_valid=false，不自动追加或删失败。\n"
                          "- 原始请求/响应、工具call ID/同批turn/真实observation、checker结果和分层统计留档；两次离线重放均通过。\n"
                          "- 置信区间按任务/场景聚类，Pass@8不是rollout平均正确率。\n"
                          "- 临时假权重只供哈希/加载合同验证，已清理；未下载/加载真实模型、未启动GPU/训练、未用test。\n"
                          "- 真实权重/后端兼容及正式预注册/test冻结仍须后续独立验证。\n").encode()
    manifest = {"version": "research-eval-CPU-verification-manifest-v1", "verifier_sha256": entry.file_hash(Path(__file__)),
                "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in files.items()}}
    files["manifest.json"] = (canonical_json(manifest) + "\n").encode()
    return files


def build_artifacts() -> dict[str, bytes]:
    return asyncio.run(build_artifacts_async())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/research/evaluation_cpu_v1")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        output = entry.checked_output(args.output)
        if output.exists() and not args.check:
            raise ValueError("output exists; refusing overwrite")
        files = build_artifacts()
        if args.check:
            actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
            if actual != set(files) or any((output / name).read_bytes() != content for name, content in files.items()):
                raise ValueError("CPU verification artifact bytes mismatch/missing/extra")
            print("PASS: all CPU mock evaluation artifact bytes reproduce; no real model/GPU")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".eval-CPU-contract-", dir=output.parent))
        try:
            for name, content in files.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            if output.exists():
                raise ValueError("output appeared; refusing overwrite")
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print("PASS: CPU mock evaluation artifacts at " + str(output.relative_to(ROOT)))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

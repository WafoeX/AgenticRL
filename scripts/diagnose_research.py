#!/usr/bin/env python3
"""CPU scripted development diagnostics; no model, GPU, reward or test input."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_research_dataset import file_hash, legacy_hashes  # noqa: E402
from src.research_answers import ANSWER_VERSION, check_answer, oracle_answer  # noqa: E402
from src.research_checker import TRACE_VERSION, check_attempt  # noqa: E402
from src.research_environment import TOOL_SCHEMAS, ResearchEnvironment, canonical_json  # noqa: E402
from src.research_generator import execute_reference  # noqa: E402
from src.research_interventions import (  # noqa: E402
    INTERVENTION_VERSION,
    build_pairs,
    check_pair,
    summarize_pairs,
)
from src.research_schema import SYSTEM_PROMPT, digest, validate_record  # noqa: E402

SOURCE_PATHS = [
    "src/research_answers.py", "src/research_checker.py", "src/research_interventions.py",
    "scripts/diagnose_research.py", "src/research_environment.py", "src/research_generator.py", "src/research_schema.py",
]


def scripted_attempt(view: dict) -> dict:
    """Observation-reading reference responder. Receives no gold/useful/expected calls."""
    trace, _, answer = execute_reference({"recipe": view["recipe_id"], "parameters": view["parameters"],
                                          "world": view["environment_snapshot"]})
    return {"content": canonical_json(answer), "trace": trace, "termination": "finished"}


def frozen_attempt(view: dict, base_attempt: dict) -> dict:
    """Reuse base arguments, but record real observations from the changed world."""
    attempt = copy.deepcopy(base_attempt)
    env = ResearchEnvironment(view["environment_snapshot"])
    for step in attempt["trace"]:
        step["observation"] = env.execute(step["name"], step["arguments"])
    return attempt


def load_validation() -> list[dict]:
    path = ROOT / "data/research_v1_candidate/validation.jsonl"
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        raise ValueError("cannot load candidate validation") from exc
    for row in records:
        validate_record(row)
        if row["split"] != "validation":
            raise ValueError("diagnostic input must be validation only")
    return records


def build_artifacts() -> dict[str, bytes]:
    before = legacy_hashes()
    records = load_validation()
    reference_results = []
    for row in records:
        attempt = scripted_attempt(row)
        result = check_attempt(row, attempt["content"], attempt["trace"])
        if not result["answer"]["correct"] or not result["trace"]["strict_complete"]:
            raise ValueError(f"reference validation failed: {row['sample_id']}")
        if not check_answer(row, canonical_json(row["oracle_answer"]))["correct"]:
            raise ValueError("saved candidate gold differs from independent checker")
        reference_results.append({"sample_id": row["sample_id"], "recipe_id": row["recipe_id"],
                                  "oracle": oracle_answer(row), "attempt": attempt, "result": result})
    pairs = build_pairs(records)
    outputs, controls = [], {name: [] for name in ("frozen_arguments", "stale_answer", "fabricated_observation", "same_turn")}
    for pair in pairs:
        a, b = scripted_attempt(pair["base"]), scripted_attempt(pair["variant"])
        result = check_pair(pair, a, b)
        if not result["responsive"] or not result["consumers_changed"]:
            raise ValueError("adaptive scripted control did not respond")
        outputs.append({"base_attempt": a, "variant_attempt": b, "result": result})
        variants = {"frozen_arguments": frozen_attempt(pair["variant"], a), "stale_answer": copy.deepcopy(b),
                    "fabricated_observation": copy.deepcopy(a), "same_turn": copy.deepcopy(b)}
        variants["stale_answer"]["content"] = a["content"]
        for step in variants["same_turn"]["trace"]:
            step["turn"] = 0
        for name, attempt in variants.items():
            checked = check_pair(pair, a, attempt)
            if checked["responsive"]:
                raise ValueError(f"negative control was accepted: {name}")
            controls[name].append({"group_id": pair["group_id"], "base_attempt": a,
                                   "variant_attempt": attempt, "result": checked})
    alternatives = []
    for pair in pairs:
        if pair["base"]["recipe_id"] != "weather_convert":
            continue
        attempts = []
        for side in ("base", "variant"):
            view = pair[side]
            args = {"city": view["parameters"]["entity"], "units": "fahrenheit"}
            observation = ResearchEnvironment(view["environment_snapshot"]).execute("get_weather", args)
            try:
                temperature = json.loads(observation)["temperature"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("invalid alternative weather observation") from exc
            attempts.append({"content": canonical_json({"value": temperature, "unit": "fahrenheit"}),
                             "trace": [{"turn": 0, "name": "get_weather", "arguments": args, "observation": observation}]})
        checked = check_pair(pair, *attempts)
        if not checked["responsive"] or not checked["alternative_bypass"] or checked["dependency_valid"]:
            raise ValueError("legal weather alternative misclassified")
        alternatives.append({"group_id": pair["group_id"], "base_attempt": attempts[0],
                             "variant_attempt": attempts[1], "result": checked})
    if before != legacy_hashes():
        raise ValueError("legacy data/tools changed")
    summary = {
        "kind": "scripted_cpu_development_diagnostic_not_model_accuracy", "test_used": False, "test_frozen": False,
        "versions": {"answer": ANSWER_VERSION, "trace": TRACE_VERSION, "intervention": INTERVENTION_VERSION},
        "references": {"planned": len(records), "answer_correct": len(reference_results), "strict_complete": len(reference_results)},
        "paired": summarize_pairs(pairs, [r["result"] for r in outputs]),
        "negative_controls": {name: {"planned_groups": len(pairs), "rejected_groups": len(rows)} for name, rows in controls.items()},
        "legal_fahrenheit_alternative_groups": len(alternatives),
        "limitations": ["Scripted responder, not model behavioral evidence or training gain.",
                        "Answer oracle independent of tools/reward/reference, but shares raw snapshot and recipe contract.",
                        "Trace replay shares ResearchEnvironment and requires trusted turn/observation provenance.",
                        "Only registered routes/commutative arithmetic recognized; answer credit is route-independent.",
                        "Validation-only grouped diagnosis, not additional independent main-accuracy samples."],
    }
    files = {"summary.json": (canonical_json(summary) + "\n").encode()}
    for name, rows in [("pairs", pairs), ("adaptive", outputs), ("references", reference_results), ("alternatives", alternatives), *controls.items()]:
        files[name + ".jsonl"] = ("".join(canonical_json(row) + "\n" for row in rows)).encode()
    count = summary["paired"]["planned_groups"]
    markdown = f"""# R1.6 / R2.1–R2.4 CPU 开发诊断

**只执行 scripted responder，不是模型能力、训练提升或冻结 test 成绩。**

- validation 参考：{len(records)}/{len(records)} 独立答案正确且严格协议完成。
- 动态配对：{count}/{count} 组，真实消费参数与答案随隐藏事实变化。
- 四种负控各 {count}/{count} 拒绝：冻结参数、旧答案、虚构 observation、同 turn 伪依赖。
- Fahrenheit 单查询合法替代：{len(alternatives)} 组，答案/配对响应正确，动态边记 bypass、不记依赖通过。
- 所有分母按源语义组保留；raw JSONL 包含原始调用、observation、最终答案和各 checker 输出。
- 未读取 test；未调用模型/GPU/reward；旧数据/工具和候选目录保持不变。

## 边界

独立 oracle 从原始快照与公开参数重算，不读取保存 gold、参考轨迹、reward 或模型 success。
协议重放仍共享工具执行器；干预诊断需要可信 turn/工具结果来源，不能凭数值吻合证明模型因果注意。
此处正控证明诊断代码能识别 scripted observation 消费，不能预告 Base/SFT/GRPO 的表现。
R2.5–R2.8 加载身份/固定模型推理入口/正式指标归档尚待完成，test 尚未冻结。
"""
    files["REPORT.md"] = markdown.encode()
    manifest = {"version": "research-diagnostic-manifest-v1", "validation_sha256": file_hash(ROOT / "data/research_v1_candidate/validation.jsonl"),
                "source_hashes": {p: file_hash(ROOT / p) for p in SOURCE_PATHS},
                "system_prompt_hash": digest(SYSTEM_PROMPT), "tool_schema_hash": digest(TOOL_SCHEMAS),
                "legacy_hashes_before": before, "legacy_hashes_after": legacy_hashes(),
                "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in files.items()}}
    files["manifest.json"] = (canonical_json(manifest) + "\n").encode()
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/research/checker_interventions_v1")
    parser.add_argument("--check", action="store_true", help="read-only full byte regeneration check")
    args = parser.parse_args()
    try:
        output = (ROOT / args.output).resolve()
        if not output.is_relative_to(ROOT / "results/research") or output == ROOT / "results/research":
            raise ValueError("output must be a dedicated results/research subdirectory")
        if output.exists() and not args.check:
            raise ValueError("output exists; refusing overwrite")
        files = build_artifacts()
        if args.check:
            if not output.is_dir() or {p.name for p in output.iterdir()} != set(files):
                raise ValueError("missing/extra diagnostic artifacts")
            for name, content in files.items():
                if (output / name).read_bytes() != content:
                    raise ValueError(f"byte mismatch: {name}")
            print("PASS: all diagnostic artifact bytes reproduce")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".diagnostic-", dir=output.parent))
        try:
            for name, content in files.items():
                (staging / name).write_bytes(content)
            if output.exists():
                raise ValueError("output appeared; refusing overwrite")
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print("PASS: CPU development diagnostic at " + str(output.relative_to(ROOT)))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

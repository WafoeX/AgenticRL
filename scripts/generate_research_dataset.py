#!/usr/bin/env python3
"""Build/check CPU-only candidate data; refuses overwrite and never freezes test."""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_audit import audit_dataset  # noqa: E402
from src.research_environment import TOOL_SCHEMAS, canonical_json  # noqa: E402
from src.research_generator import generate_dataset  # noqa: E402
from src.research_schema import SYSTEM_PROMPT, digest, load_json  # noqa: E402

SOURCE_PATHS = [
    "src/research_environment.py", "src/research_schema.py", "src/research_generator.py", "src/research_audit.py",
    "scripts/generate_research_dataset.py", "configs/research/dataset_v1.json", "configs/research/schema_v1.json",
]
LEGACY_PATHS = ["data/scenarios_train.jsonl", "data/scenarios_val.jsonl", "src/tools.py", "src/scenarios.py"]


def file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash file: {path}") from exc


def legacy_hashes() -> dict:
    return {p: file_hash(ROOT / p) for p in LEGACY_PATHS}


def source_identity() -> dict:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError("cannot identify source commit")
    hashes = {p: file_hash(ROOT / p) for p in SOURCE_PATHS}
    tracked = subprocess.run(["git", "ls-files", "--", *SOURCE_PATHS], cwd=ROOT, text=True, capture_output=True, check=False)
    if tracked.returncode:
        raise ValueError("cannot identify tracked source paths")
    return {"base_commit": result.stdout.strip(), "working_tree_file_hashes": hashes,
            "source_hash": digest(hashes), "untracked_source_files": sorted(set(SOURCE_PATHS) - set(tracked.stdout.splitlines())),
            "note": "Working-tree hashes identify this implementation; base_commit alone may not contain it."}


def markdown_report(audit: dict, allocation: dict) -> str:
    lines = ["# Research V1 候选数据 CPU 验收", "", "**状态：构建/容量/泄漏/参考执行通过；test 尚未冻结，无 GPU 或模型评测。**", "",
             "## 数据计数", "", "| split | 总数 | 1-call | 2-call | 3-call | 动态依赖 |", "|---|---:|---:|---:|---:|---:|"]
    for split, summary in audit["datasets"].items():
        depth = summary["depth_counts"]
        lines.append(f"| {split} | {summary['rows']} | {depth.get('1', 0)} | {depth.get('2', 0)} | {depth.get('3', 0)} | {summary['dynamic_dependency_rows']} |")
    lines += ["", "## 泛化分层", "", "| split | 分层 | 数量 |", "|---|---|---:|"]
    for split, summary in audit["datasets"].items():
        for kind, count in summary["generalization_counts"].items():
            lines.append(f"| {split} | {kind} | {count} |")
    lines += ["", "## 容量与泄漏", "", f"逐个枚举 {len(allocation['capacity'])} 个 recipe/domain 池，每池最多1024候选；按去实体名的值/操作签名计数。",
              f"最紧需求池剩余下界：{min(c['remaining_lower_bound'] for c in allocation['capacity'].values())}。这是注册有限池容量，不是所有理论世界的容量。",
              "全局先分组再分配，不删除已输出样本；候选冲突只计分配重试，无法填满配额直接失败。",
              f"分配候选冲突计数：`{canonical_json(allocation['allocation_rejections'])}`。", "",
              "跨 split 精确文本、规范化文本、语义组、去实体名值签名、规范化完整轨迹签名重叠全部为0；split内部同样无重复。",
              f"全部 {audit['replayed_references']} 条参考已重放并与独立 Decimal 公式核对。JSON审计含每类配额、比较A/B平衡和组合序列是否在train出现。", "",
              "## 边界", "", "- 生成的oracle仅用于参考构建验收；尚不是R2的最终模型答案解析/判定器。",
              "- 不包含R1.6干预诊断、不冻结test、不接入旧rollout/SFT/reward；旧数据及工具hash在manifest中留档。",
              "- 实体来自共享命名池，但部分holdout实体可能未在train出现；audit.json单列实体新颖性，不能声称纯数值单因素OOD。",
              "- 两比较recipe在每个分层中A/B平衡；正式候选集无tie，CPU fixtures测试tie。",
              "- weather_convert的一调用Fahrenheit替代解合法；参考深度不等于必要调用次数。",
              "- 三步基础任务只有密度转换，家族/深度存在混杂；不能把全局深度曲线当通用多步能力。",
              "- 合成快照共享事实输入，不是外部盲测；checker/reward/protocol未锁定前不得宣称冻结测试。", "",
              "## 训练 observation 的事实暴露诊断", "",
              "允许共享环境事实不等于等价任务标签泄漏，但可能存在记忆捷径；只统计真实train工具返回过的字段，不统计未查询的隐藏快照。",
              "匿名实体来自与参数索引无直接可见关系的固定hash映射；模型prompt不包含生成seed/index。", "",
              "| split | 分层 | 需隐藏事实的任务 | 至少一项train见过 | 全部train见过 |", "|---|---|---:|---:|---:|"]
    for split, strata in audit["train_observation_fact_exposure"].items():
        for kind, stats in strata.items():
            lines.append(f"| {split} | {kind} | {stats['rows_requiring_hidden_facts']} | {stats['rows_with_any_fact_previously_observed_in_train']} | {stats['rows_with_all_facts_previously_observed_in_train']} |")
    lines.append("")
    return "\n".join(lines)


def build_artifacts(config: dict, before: dict) -> tuple[dict[str, bytes], dict]:
    data, allocation = generate_dataset(config)
    audit = audit_dataset(data, config)
    audit["allocation"] = allocation
    files = {
        f"{split}.jsonl": ("\n".join(canonical_json(row) for row in rows) + "\n").encode()
        for split, rows in data.items()
    }
    files["audit.json"] = (canonical_json(audit) + "\n").encode()
    files["DATA_CARD.md"] = markdown_report(audit, allocation).encode()
    after = legacy_hashes()
    if before != after:
        raise ValueError("legacy data/tools changed during candidate construction")
    manifest = {
        "manifest_version": "research-candidate-manifest-v1", "dataset_version": config["dataset_version"],
        "seed": config["seed"], "test_status": "candidate_not_frozen", "config_hash": digest(config),
        "source": source_identity(), "tool_schema_hash": digest(TOOL_SCHEMAS), "system_prompt_hash": digest(SYSTEM_PROMPT),
        "legacy_hashes_before": before, "legacy_hashes_after": after,
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in files.items()},
        "counts": {split: len(rows) for split, rows in data.items()},
        "reproducibility": "No clocks/global RNG/Python hash; --check re-generates and compares all artifact bytes, including manifest.",
    }
    files["manifest.json"] = (canonical_json(manifest) + "\n").encode()
    return files, manifest


def checked_output(value: str) -> Path:
    output = Path(value)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    allowed = ROOT / "data/research_v1_candidate"
    research_results = ROOT / "results/research"
    if output != allowed and not output.is_relative_to(research_results):
        raise ValueError("output must be data/research_v1_candidate or a child of results/research")
    if output == research_results:
        raise ValueError("output must be a dedicated results subdirectory")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/dataset_v1.json")
    parser.add_argument("--output", default="data/research_v1_candidate")
    parser.add_argument("--check", action="store_true", help="read-only full regeneration + byte comparison; no writes")
    args = parser.parse_args()
    try:
        output = checked_output(args.output)
        config_path = Path(args.config)
        config = load_json(config_path if config_path.is_absolute() else ROOT / config_path)
        if not args.check and output.exists():
            raise ValueError("output already exists; refusing overwrite (use --check or a new results directory)")
        before = legacy_hashes()
        files, manifest = build_artifacts(config, before)
        if args.check:
            if not output.is_dir() or {p.name for p in output.iterdir()} != set(files):
                raise ValueError("artifact files missing/extra")
            for name, content in files.items():
                if (output / name).read_bytes() != content:
                    raise ValueError(f"regeneration byte mismatch: {name}")
            print("PASS: all artifact bytes reproduce; counts=" + canonical_json(manifest["counts"]))
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".research-build-", dir=output.parent))
        try:
            for name, content in files.items():
                (staging / name).write_bytes(content)
            if output.exists():
                raise ValueError("output appeared during construction; refusing overwrite")
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print("PASS: candidate artifacts at " + str(output.relative_to(ROOT)))
        print("counts=" + canonical_json(manifest["counts"]) + "; test is NOT frozen")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

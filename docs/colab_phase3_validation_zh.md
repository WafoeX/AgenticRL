# Colab A100：Phase 3 验证步骤与实验命令

目标：在 GPU 上验收 Reward V2、完整工具链判定及同轨迹双评分。**先测试，再审计；本轮不做 10/50-step 训练，不推进 Phase 4。** 本文命令由用户在 Colab 执行，尚无此环境下的运行结果。

## 0. 开始前的选择

在 Colab 的「运行时 → 更改运行时类型」中选择 GPU / A100，并实际检查设备名称。A100 算力不代表 ART、PyTorch、CUDA 和 vLLM 版本自然兼容。

选择下面一种 checkpoint 路径：

- **A：有原服务器的 step 2。** 将完整 `.art` 备份保存到 Drive，再按第 5A 节恢复。此路径适合审计已有模型，但跨环境采样结果不要求与此前百分比完全一致。
- **B：没有原 checkpoint。** 按第 5B 节在全新隔离目录做 V1 的 2-step smoke test，再验收。得到的是**新的实验模型**，不是此前 step 2 的精确复现。

不要把上游作者的 Hugging Face 合并模型当成原服务器 ART checkpoint。代码仓库不含 `.art`；仅 `git clone` 不会恢复训练权重。

**不要逐格运行旧 `notebooks/train_colab*.ipynb`：其中有 `rm -rf .art`、数据重生成及直接启动 50-step 训练的历史流程。以下各个代码块分别粘贴到 Colab 单元格；`%%bash` 必须是单元格第一行。**

## 1. 检查 GPU、挂载 Drive、设置路径

Python 单元格：

```python
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive

assert sys.version_info >= (3, 11), "需要 Python 3.11+"
subprocess.run(["nvidia-smi"], check=True)
drive.mount("/content/drive")

repo = Path("/content/AgenticRL")
run_dir = Path("/content/drive/MyDrive/AgenticRL-validation") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_dir.mkdir(parents=True, exist_ok=False)
os.environ["AGENT_REPO"] = str(repo)
os.environ["AGENT_PY"] = "/content/agent-venv/bin/python"
os.environ["AGENT_RUN"] = str(run_dir)
print("日志与结果目录：", run_dir)
```

后续 shell 单元格从 notebook 继承这三个环境变量。重启 notebook kernel 后，应重新设置变量，继续使用已创建的结果目录，不要误创建或切换实验身份。

## 2. 在 Colab/服务器上通过 SSH 拉取仓库

你已配置 GitHub SSH，以下直接使用 `git@github.com:WafoeX/AgenticRL.git`，不再使用 HTTPS Token 或 Colab `GITHUB_TOKEN`。SSH key 需要该仓库的读取权限；第 9 节推送反馈还需要写入权限（只读 deploy key 不够）。不要提交私钥、SSH 配置或 token。

先确认当前 notebook 启动的 shell 能使用你的密钥，并核对首次连接的 GitHub 主机指纹；不要设置 `StrictHostKeyChecking=no`：

```bash
%%bash
set -euo pipefail
git ls-remote git@github.com:WafoeX/AgenticRL.git refs/heads/master
```

全新目录的克隆命令：

```bash
%%bash
set -euo pipefail
test ! -e "$AGENT_REPO" || { echo "目录已存在：不要删除，请使用下面的更新步骤"; exit 1; }
git clone --branch master --single-branch \
  git@github.com:WafoeX/AgenticRL.git "$AGENT_REPO"
```

如果目录已存在，**二选一**使用下面的更新命令，不再运行上面的克隆：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
test -z "$(git status --porcelain)" || { echo "有未提交改动：先保存，禁止覆盖"; exit 1; }
test "$(git branch --show-current)" = master || { echo "当前不是 master，先核对工作目录"; exit 1; }
# 若以前通过 HTTPS 克隆，切换为 SSH；不改变训练身份或 .art
git remote set-url origin git@github.com:WafoeX/AgenticRL.git
git pull --ff-only origin master
```

先单独备份忽略目录 `.art`；上面的 clean 检查不会覆盖检查被 Git 忽略的训练状态。若 22 端口被网络封锁，可先测试 GitHub SSH 443：

```bash
%%bash
set -euo pipefail
git -c core.sshCommand='ssh -o HostName=ssh.github.com -p 443 -o HostKeyAlias=github.com' \
  ls-remote git@github.com:WafoeX/AgenticRL.git refs/heads/master
```

该调用只临时改变 SSH 连接方式；成功后，对需要的 `clone` / `pull` / `fetch` / `push` 同样添加此 `-c` 参数，或在服务器已有的 SSH 配置中设置相应连接。若使用 ssh-agent，须确保 `SSH_AUTH_SOCK` 能被这些单元格继承；仅在另一个短生命周期 shell 启动 agent 不一定可用。

完成克隆/更新后记录实验来源：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
git rev-parse HEAD | tee "$AGENT_RUN/source_commit.txt"
git status --short | tee "$AGENT_RUN/git_status_initial.txt"
sha256sum data/scenarios_train.jsonl data/scenarios_val.jsonl configs/train_config.yaml \
  | tee "$AGENT_RUN/input_sha256.txt"
```

核对 `source_commit.txt` 与本次推送提交相同。已有 checkout 若要更新，先备份 `.art` 和未提交改动，再使用 `git pull --ff-only`；不要执行重置或清空目录。

## 3. 使用隔离 Python 环境安装依赖

避免原地替换 Colab notebook kernel 的 torch。后续项目命令全部使用 **`$AGENT_PY`**，而不是裸 `python`。

```python
%pip install -q uv
```

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
export UV_HTTP_TIMEOUT=300
uv venv --python python3 /content/agent-venv
uv pip install --python "$AGENT_PY" -r requirements.txt \
  2>&1 | tee "$AGENT_RUN/install.log"
uv pip check --python "$AGENT_PY" 2>&1 | tee "$AGENT_RUN/dependency_check.log"
uv pip freeze --python "$AGENT_PY" > "$AGENT_RUN/packages.txt"
nvidia-smi > "$AGENT_RUN/nvidia-smi.txt"
```

`set -o pipefail` 保证命令失败不会被 `tee` 的成功掩盖。若依赖检查失败，先排查日志，不要通过无序升降级把环境变成另一套实验。

验证真实后端导入和 GPU：

```bash
%%bash
set -euo pipefail
"$AGENT_PY" -u - <<'PY' 2>&1 | tee "$AGENT_RUN/backend_preflight.log"
import sys
import importlib.metadata as md
import unsloth  # 在 transformers 等相关组件之前导入
import torch
import art
import vllm
import trl
from art.local.backend import LocalBackend

print("Python:", sys.version)
for name in ("openpipe-art", "torch", "vllm", "trl", "transformers", "unsloth", "unsloth-zoo"):
    print(name, md.version(name))
print("Torch CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "隔离环境无法使用 CUDA：先解决驱动/torch 兼容问题"
print("GPU:", torch.cuda.get_device_name(0))
assert "A100" in torch.cuda.get_device_name(0), "本次计划使用 A100，请确认分配的设备"
print("LocalBackend import OK")
PY
```

**兼容性注意：**历史文档写的是定制 `vllm-0.17.0+art1`；ART `v0.5.17` 官方 tag 的依赖声明与此不同。不要仅凭版本字符串决定强制重装，更不能保证当前 PyPI 安装一定复现历史环境。以本次解析出的包、GPU driver 和实际 `LocalBackend` 导入/运行结果为准。安装或导入失败时，返回 `install.log` 和 `backend_preflight.log`，先停止 GPU 实验。

`patch_art_tokenize.py` 是历史补丁，不是每次必跑步骤。仅在实际遇到相应的 BatchEncoding / `.index()` 兼容错误、核对补丁适用后，在此隔离环境执行：

```bash
# Colab 中放入单独的 %%bash 单元格；仅在匹配该已知错误时使用
cd "$AGENT_REPO"
"$AGENT_PY" scripts/patch_art_tokenize.py
```

补丁修改安装包；若使用，务必记录命令、输出与修改后文件哈希，不冒称未修改的 ART 环境。

## 4. 先运行全套测试

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
"$AGENT_PY" -u -m pytest tests/ -v \
  2>&1 | tee "$AGENT_RUN/pytest.log"
```

要求：全部通过，无失败/错误。不预设测试数量；记录本次真实 summary。测试中包含替身，**测试通过不等于 GPU 推理已验收**。未通过则先停在此处，不启动训练或评估。

## 5A. 有原服务器 step 2：恢复工作副本（优先路径）

从原服务器备份完整 `.art`，保留模型身份与 ART 状态，不仅复制单个 LoRA 文件。假设你上传到 Drive 后，目录结构是：

```text
/content/drive/MyDrive/AgenticRL-checkpoints/server-step2/.art/...
```

Colab 中执行：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
SRC="/content/drive/MyDrive/AgenticRL-checkpoints/server-step2/.art"
test -d "$SRC" || { echo "未找到 step 2 备份，请修正 SRC"; exit 1; }
test ! -e .art || { echo "当前 .art 已存在：停止，避免覆盖"; exit 1; }
cp -a "$SRC" .art
```

Drive 原件不作为训练工作目录。ART 元数据/路径可能受跨机器迁移影响；后续日志应实际显示 **`Loaded checkpoint: step 2`**。若显示 step 0、其他步数或加载失败，不得把这次结果称为原模型复现；先排查路径、ART 状态和配置中的 `model.name` / `model.project`。

保持 `AGENT_REPO=/content/AgenticRL`，跳到第 6 节；**不要运行第 5B 节训练。**

## 5B. 没有原 checkpoint：隔离目录做 V1 两步 smoke test（可选路径）

这是新训练，只用于验证基础链路。训练函数会调用 `model.delete_checkpoints()`，因此**绝不能在原 checkpoint 工作目录运行它**。

在新 worktree 中执行，避免任何 `.art` 沿用：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
test ! -e /content/AgenticRL-smoke || { echo "smoke 目录已存在，停止避免重用状态"; exit 1; }
git worktree add --detach /content/AgenticRL-smoke "$(git rev-parse HEAD)"
cd /content/AgenticRL-smoke
test ! -e .art
"$AGENT_PY" -u -m src.train \
  --config configs/train_config.yaml \
  --steps 2 --rollouts 32 --reward-version v1 \
  2>&1 | tee "$AGENT_RUN/train_smoke_v1.log"
test -d .art
tar -czf "$AGENT_RUN/art_smoke_after_train.tar.gz" .art
```

核对 `Starting from step 0`、每步 4 组 × 8 条轨迹，以及真实梯度更新/保存记录；只看到程序打印 `Training complete` 不足以证明更新执行了，持续 SKIP 时先排查。

`--steps 2` 是目标总步数，不是“在已有 checkpoint 上再训练两步”。此处 `--rollouts 32` 仅用于新 smoke；数据保持仓库原文件，不重新生成。

训练成功后，用 Python 单元格切换评估目录：

```python
import os
os.environ["AGENT_REPO"] = "/content/AgenticRL-smoke"
print("后续评估使用新的 smoke checkpoint，不是原服务器权重")
```

后续文件中的“step2”只指这个新 smoke 的步骤；报告中注明来源，不能与旧实验直接计算学习增益。

## 6. 实验一：V1 双模式 + 完整链路 + 同 trace 双评分

下面是本轮主要审计命令。模型不更新，V1 为主 reward；同时对每条 trace 计算 V1/V2，并保存原始工具执行记录。

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
"$AGENT_PY" -u scripts/eval_checkpoint.py \
  --config configs/train_config.yaml \
  --scenarios data/scenarios_val.jsonl \
  --mode both --generations 8 --batch-scenarios 4 \
  --reward-version v1 \
  --complete-chain --compare-rewards --save-traces \
  --output "$AGENT_RUN/eval_step2_reward_audit.json" \
  2>&1 | tee "$AGENT_RUN/eval_step2_reward_audit.log"
```

要求：同一模型顺序完成 **50 条 deterministic + 400 条 sample**；核对加载步数和失败率。基础设施失败不应被当成工具链成功，双评分的排除数必须明确记录。

## 7. 实验二：V2 确定性 rollout 集成

仍使用相同 checkpoint，不训练，验证 V2 reward 能正确进入 ART metrics 和报告：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
"$AGENT_PY" -u scripts/eval_checkpoint.py \
  --config configs/train_config.yaml \
  --scenarios data/scenarios_val.jsonl \
  --mode deterministic --batch-scenarios 4 \
  --reward-version v2 \
  --complete-chain --compare-rewards --save-traces \
  --output "$AGENT_RUN/eval_step2_v2_deterministic.json" \
  2>&1 | tee "$AGENT_RUN/eval_step2_v2_deterministic.log"
```

要求：50 条轨迹，V2 主评分与同 trace V2 comparison 一致；V1 主评分运行中的 V1 comparison 也应一致。不同运行不要求产生逐字节相同回复；用**同一次运行、同一条 trace**核对评分。

若两种实验有基础设施失败，保留失败日志和报告。可以将 `--batch-scenarios 4` 改成 `1`，使用**新输出文件名**重跑，并记录并发设置；不要覆盖第一次失败证据，也不把不同设置当作严格同条件比较。

## 8. 汇总检查、留档与反馈

此单元格只读取结果 JSON，不执行模型：

```python
import os
import json
import math
from pathlib import Path

run_dir = Path(os.environ["AGENT_RUN"])
audit = json.loads((run_dir / "eval_step2_reward_audit.json").read_text())
v2 = json.loads((run_dir / "eval_step2_v2_deterministic.json").read_text())
reports = {
    "v1_deterministic": audit["evaluations"]["deterministic"],
    "v1_sample": audit["evaluations"]["sample"],
    "v2_deterministic": v2,
}
summary = {}
for label, report in reports.items():
    expected_attempts = 400 if label == "v1_sample" else 50
    assert report["step"] == 2, f"{label}: 实际加载步数不是2"
    assert report["total_scenarios"] == 50
    assert report["total_rollouts"] == expected_attempts
    attempts = [a for s in report["per_scenario"] for a in s["attempts"]]
    assert len(attempts) == expected_attempts
    scorer = report["reward_version"]
    for attempt in attempts:
        comparison = attempt["reward_comparison"]
        if comparison["excluded_reason"] is not None:
            continue  # 排除数保留在报告中，不冒充成功
        assert math.isclose(attempt["reward"], comparison[scorer], abs_tol=1e-9)
        if not attempt["complete_tool_chain"]:
            assert comparison["v2"] <= 2.0 + 1e-9
    summary[label] = {
        "step": report["step"],
        "reward_version": scorer,
        "total_rollouts": report["total_rollouts"],
        "infra_failure_rate": report["infra_failure_rate"],
        "reward_threshold_success_pct": report.get("pass_at_k_pct", report.get("deterministic_accuracy_pct")),
        "complete_chain": report["complete_chain"]["overall"],
        "reward_comparison": report["reward_comparison"],
    }

(run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("以上是结构与同轨迹评分检查，不代替人工审阅动态依赖和失败原因。")
```

人工重点检查：

1. 查看 `country_compare` 实际 trace：只做一次 lookup 时，完整链路必须为 false，V2 ≤2。
2. 数学参数的空格/括号变化应经 normalization 正确评分；不要仅凭 reward=2 推断失败来自空格。
3. 动态参数须来自正确的前一步 observation，依赖消费 turn 严格晚于生产 turn；未支持路由不给动态 credit。
4. `infra_failure_rate` 非零、后端监控异常或持续训练 SKIP 需要说明；不能仅凭最终汇总宣称链路完全稳定。
5. V2 分数变化只说明评分器不同，不代表模型经过 V2 训练或能力改善。

最后保存运行结束时的源码状态；如有新 smoke 状态或本地恢复的 checkpoint，需要备份时另存归档，不覆盖 Drive 中的原件：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
git status --short > "$AGENT_RUN/git_status_final.txt"
sha256sum data/scenarios_train.jsonl data/scenarios_val.jsonl configs/train_config.yaml \
  > "$AGENT_RUN/input_sha256_final.txt"
```

请反馈 `source_commit.txt`、`packages.txt`、`nvidia-smi.txt`、`pytest.log`、两个评估 JSON/日志及 `summary.json`，并注明使用 **A 原 checkpoint** 还是 **B 新 smoke**。结果已写入 Drive；第 9 节将小型结果通过 SSH 推送到反馈分支，大文件仍保留原件。

**完成标准：**当前测试全部通过；后端实际可运行；checkpoint/数据/代码身份明确；两项 GPU 审计完整返回，评分与 trace 对齐；失败与人工审阅结论均有记录。完成后再判断是否验收 Phase 3，不能预先更新为“全部通过”。

## 9. 通过 SSH 推送反馈；大文件另行下载

建议每次实验使用独立 `feedback/<run-id>` 分支，不把结果提交混入训练源码分支，也不在 detached smoke worktree 上直接提交。结果目录固定为 `results/phase3/<run-id>/`；读取时以其中的 `source_commit.txt` 为实验代码身份，而不是反馈提交的 HEAD。

默认仅回传明确列出的日志、环境、摘要和评估 JSON。**单文件超过 10 MiB，或累计复制量将超过 50 MiB 时，先只提交清单，不强行上传**（这是建议的回传预算，不是 GitHub 的硬限制）。不提交 `.art`、checkpoint 压缩包、模型权重、SSH key 或 token；不要使用 `git add .`。失败运行也可以回传已有日志，不必为了产生 summary 伪造成功。

### 9.1 创建独立反馈 worktree

实验结束后执行。即使 `AGENT_REPO` 已切换到 smoke 目录，仍使用第 2 节的主 checkout 创建反馈分支，不改动实验工作目录。

```bash
%%bash
set -euo pipefail
BASE=/content/AgenticRL
RUN_ID="$(basename "$AGENT_RUN")"
BRANCH="feedback/$RUN_ID"
FEEDBACK_DIR="/content/AgenticRL-feedback-$RUN_ID"
test ! -e "$FEEDBACK_DIR" || { echo "反馈目录已存在：先核对，不要删除或重复创建"; exit 1; }
git -C "$BASE" fetch origin master
git -C "$BASE" worktree add -b "$BRANCH" "$FEEDBACK_DIR" origin/master
```

### 9.2 收集小文件并生成清单

Python 单元格：**先将 `CHECKPOINT_SOURCE` 改为你实际使用的 A/B 路径，并填写备注**。脚本只读取结果，不重新跑模型；未知文件和大文件保留在 Drive，并在 manifest 中标明。

```python
import os
import json
import hashlib
import shutil
from pathlib import Path

CHECKPOINT_SOURCE = "A-original-server-step2"  # 若使用新 smoke，改为 B-new-colab-smoke
NOTES = "填写人工审阅结论、失败/重跑/补丁情况；不要填入密钥或token"
run_dir = Path(os.environ["AGENT_RUN"])
run_id = run_dir.name
feedback_dir = Path("/content") / f"AgenticRL-feedback-{run_id}"
assert (feedback_dir / ".git").exists(), "先执行9.1创建反馈worktree"
out = feedback_dir / "results" / "phase3" / run_id
out.mkdir(parents=True, exist_ok=False)

allowed = {
    "source_commit.txt", "git_status_initial.txt", "git_status_final.txt",
    "input_sha256.txt", "input_sha256_final.txt", "packages.txt", "nvidia-smi.txt",
    "install.log", "dependency_check.log", "backend_preflight.log", "pytest.log",
    "train_smoke_v1.log", "eval_step2_reward_audit.json", "eval_step2_reward_audit.log",
    "eval_step2_v2_deterministic.json", "eval_step2_v2_deterministic.log", "summary.json",
}
per_file_limit = 10 * 1024 * 1024
budget = 50 * 1024 * 1024
used = 0
artifacts = []
for src in sorted(run_dir.iterdir()):
    if src.is_symlink() or not src.is_file():
        continue
    size = src.stat().st_size
    digest = hashlib.sha256()
    with src.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if src.name not in allowed:
        reason = "not-in-upload-allowlist"
    elif size > per_file_limit:
        reason = "over-per-file-limit"
    elif used + size > budget:
        reason = "over-total-budget"
    else:
        reason = None
        shutil.copy2(src, out / src.name)
        used += size
    artifacts.append({
        "name": src.name, "bytes": size, "sha256": digest.hexdigest(),
        "copied_to_git": reason is None, "excluded_reason": reason,
    })

manifest = {
    "run_id": run_id,
    "checkpoint_source": CHECKPOINT_SOURCE,
    "notes": NOTES,
    "artifact_location": f"Google Drive: MyDrive/AgenticRL-validation/{run_id}",
    "copied_bytes": used,
    "artifacts": artifacts,
    "missing_expected_files": sorted(allowed - {p.name for p in run_dir.iterdir() if p.is_file()}),
}
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("反馈目录：", out)
print("准备提交字节数：", used)
print("请先人工检查日志无凭据，再执行9.3。")
```

完整扫描大文件计算 SHA-256 可能耗时；原文件不被删改。`missing_expected_files` 表示缺失记录，不自动判定失败（例如 A 路径没有 smoke 日志）；实际是否验收要结合日志判断。若使用了不同输出名的重跑，手工将对应小文件纳入白名单，并记录新命令，不覆盖首次证据。

### 9.3 检查后提交、推送反馈分支

**先人工检查 manifest、日志和 JSON 不含凭据。** Git 提交身份须已配置；`git var GIT_AUTHOR_IDENT` 若报错，先用 `git config user.name` / `git config user.email` 设置你自己的身份。

```bash
%%bash
set -euo pipefail
RUN_ID="$(basename "$AGENT_RUN")"
BRANCH="feedback/$RUN_ID"
cd "/content/AgenticRL-feedback-$RUN_ID"
git var GIT_AUTHOR_IDENT >/dev/null
git status --short
git add -- "results/phase3/$RUN_ID"
git diff --cached --check
git diff --cached --stat
git commit -m "results: record Colab Phase 3 validation $RUN_ID"
git push -u origin "$BRANCH"
printf '反馈分支：%s\n反馈提交：' "$BRANCH"
git rev-parse HEAD
```

推送被拒绝时先检查写入权限；不使用 force push。22 端口受限时，按第 2 节的 443 方式调用 fetch/push。

### 9.4 我在本地读取与分析的流程

推送后在聊天中告诉我：**分支名、反馈提交号、A/B checkpoint 来源**。我会通过 SSH fetch 该分支，在独立 worktree 中读取结果，不把反馈提交自动合并到开发分支：

```bash
# 以下由本地分析端执行；将 RUN_ID 替换为你反馈的真实值
RUN_ID=实际运行ID
git fetch origin "refs/heads/feedback/$RUN_ID:refs/remotes/origin/feedback/$RUN_ID"
git worktree add --detach "../AgenticRL-analysis-$RUN_ID" "origin/feedback/$RUN_ID"
```

先核对 manifest、SHA-256、代码/数据身份、测试摘要，再分析原始评估分母、infra、完整链路与同 trace 分项。不只看 reward 上升，也不把不同 checkpoint 或不同采样策略混作学习增益。

若关键评估 JSON/trace 太大未进入 Git，你再从 Drive 下载对应文件，提供本地路径或可访问的下载位置，我据 manifest 的文件名、大小、SHA-256 核对后继续分析。仅提交摘要不能代替缺失 trace 的审计，也不应据此宣称 Phase 3 全部验收。

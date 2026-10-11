# Colab：SSH 拉取修复后进行 2-step 训练与退出验证

这是**新的 V1 smoke 实验**，不是被删除运行时中原 step 2 的复现。默认训练目标 2 steps，每步 32 rollouts（4 组 × 8）；先检查训练退出，再做单场景 deterministic 评估，最后可选全量审计。不启动 50-step 训练，不重新生成数据，不删除已有 `.art`。

如果你只想验证已有 checkpoint 的评估退出，不需要重训，使用 [已有 checkpoint 最小检查](backend_lifecycle_fix_zh.md)。以下按你本次要求提供新训练测试流程。

**按顺序把各代码块放入独立 Colab 单元格。`%%bash` 必须是该单元格第一行。** 推荐与上一轮一样的 A100 GPU。SSH 密钥由你配置，不将密钥/token写入代码、日志或 Git。

## 1. 挂载 Drive，设定本次独立目录

Python 单元格：

```python
import os
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive

drive.mount('/content/drive')
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run = Path('/content/drive/MyDrive/AgenticRL-exit-validation') / run_id
run.mkdir(parents=True, exist_ok=False)
os.environ.update(
    AGENT_REPO='/content/AgenticRL',
    AGENT_PY='/content/agent-exit-venv/bin/python',
    AGENT_RUN=str(run),
    AGENT_SMOKE='/content/AgenticRL-exit-smoke-' + run_id,
    WANDB_MODE='disabled',
    PYTHONUNBUFFERED='1',
)
print('本次日志/备份：', run)
print('独立训练目录：', os.environ['AGENT_SMOKE'])
```

不要在后续单元格重复执行这一步而无意换了实验目录。内核重启后须恢复这些变量，不能沿用已删除运行时的路径。

## 2. 通过 SSH 拉取 master

```bash
%%bash
set -euo pipefail
# 如果当前网络的 22 端口不可用，可取消下一行注释，改用 GitHub SSH 443。
# export GIT_SSH_COMMAND='ssh -o HostName=ssh.github.com -p 443 -o HostKeyAlias=github.com'

if [ ! -e "$AGENT_REPO" ]; then
  git clone --branch master --single-branch \
    git@github.com:WafoeX/AgenticRL.git "$AGENT_REPO"
else
  cd "$AGENT_REPO"
  test "$(git branch --show-current)" = master
  test -z "$(git status --porcelain)" || { echo '有未提交改动：停止，不覆盖'; exit 1; }
  git remote set-url origin git@github.com:WafoeX/AgenticRL.git
  git pull --ff-only origin master
fi
cd "$AGENT_REPO"
test -f src/backend_lifecycle.py
printf 'code_commit='; git rev-parse HEAD
git rev-parse HEAD > "$AGENT_RUN/source_commit.txt"
printf 'B-new-colab-lifecycle-smoke\n' > "$AGENT_RUN/checkpoint_source.txt"
sha256sum data/scenarios_train.jsonl data/scenarios_val.jsonl configs/train_config.yaml \
  > "$AGENT_RUN/input_sha256.txt"
```

保留 SSH 主机指纹校验，不设置 `StrictHostKeyChecking=no`。旧 checkout 必须拉到含 `src/backend_lifecycle.py` 的修复版本。

## 3. 安装隔离环境，沿用已验证过的 PEFT 版本

先执行 Python 单元格：

```python
%pip install -q uv
```

再执行：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
export UV_HTTP_TIMEOUT=300
# 新运行时第一次执行；若目录已经存在，先确认不是另一份实验环境，不要删除它。
test ! -e /content/agent-exit-venv || { echo 'venv 已存在：核对后从检查步骤继续'; exit 1; }
uv venv --python python3 --seed /content/agent-exit-venv
uv pip install --python "$AGENT_PY" -r requirements.txt \
  > "$AGENT_RUN/install.log" 2>&1
uv pip freeze --python "$AGENT_PY" > "$AGENT_RUN/packages_before_peft_fix.txt"
# 你上一轮成功训练使用 PEFT 0.18.0，避免再次沿用触发初始化错误的版本。
# 仅替换 PEFT，不连带升降级 torch/transformers/Unsloth。
uv pip install --python "$AGENT_PY" --no-deps 'peft==0.18.0' \
  > "$AGENT_RUN/peft_fix_install.log" 2>&1
uv pip check --python "$AGENT_PY" > "$AGENT_RUN/dependency_check.log" 2>&1
uv pip freeze --python "$AGENT_PY" > "$AGENT_RUN/packages.txt"
echo '依赖安装及一致性检查通过；详情见 Drive 日志'
```

PEFT 0.18.0 是沿用你上一轮实际成功的处理，不代表已复原上一轮完整 freeze，也不保证重新解析的其他包在新 Colab 上兼容。如果安装或 dependency check 失败，**停在这里并提供日志**，不要继续训练、不盲目升级整套依赖。

历史 `patch_art_tokenize.py` 不是必跑步骤；只在遇到匹配的 BatchEncoding / `.index()` 错误时核对适用性再使用，记录修改。

## 4. 后端/GPU检查与全套单元测试

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
nvidia-smi > "$AGENT_RUN/nvidia-smi.txt"
"$AGENT_PY" -u - <<'PY' > "$AGENT_RUN/backend_preflight.log" 2>&1
import sys
import importlib.metadata as md
import unsloth  # 必须在 transformers 等相关组件之前导入
import torch
from art.local.backend import LocalBackend
from src.backend_lifecycle import managed_backend
print('Python:', sys.version)
for package in ('openpipe-art', 'torch', 'vllm', 'trl', 'transformers',
                'unsloth', 'unsloth-zoo', 'peft', 'torchao', 'psutil'):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, 'not installed')
assert md.version('openpipe-art') == '0.5.17'
assert md.version('peft') == '0.18.0'
assert torch.cuda.is_available(), 'CUDA 不可用，停止 GPU 测试'
print('Torch CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('Backend and lifecycle imports OK')
PY
"$AGENT_PY" -m pytest tests/ -q > "$AGENT_RUN/pytest.log" 2>&1
cat "$AGENT_RUN/backend_preflight.log"
cat "$AGENT_RUN/pytest.log"
```

当前版本本地为 **173 passed**，其中 23 项生命周期回归。Colab 必须真实全部通过后才继续；这些 CPU 测试不等于 GPU 生命周期验收。

## 5. 在独立 worktree 做 V1 2-step smoke，并检查训练退出

这里不用 `tee` 管道：日志直接写入 Drive 文件，避免后代持有 stdout 管道导致 notebook 等待 EOF，同时用 Python 的 `-u` 保证及时写日志。单元格运行期间可在 Drive 中查看 `train_smoke_v1.log`。

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
test ! -e "$AGENT_SMOKE" || { echo 'smoke 目录已存在：停止，禁止沿用旧状态'; exit 1; }
git worktree add --detach "$AGENT_SMOKE" "$(git rev-parse HEAD)"
cd "$AGENT_SMOKE"
test ! -e .art
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$AGENT_RUN/gpu_before_train.csv"
echo "开始 2-step 训练。日志：$AGENT_RUN/train_smoke_v1.log"
set +e
timeout --signal=INT --kill-after=30s 20m \
  "$AGENT_PY" -u -m src.train \
    --config configs/train_config.yaml \
    --steps 2 --rollouts 32 --reward-version v1 \
    > "$AGENT_RUN/train_smoke_v1.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$AGENT_RUN/train_exit_code.txt"
printf 'train_exit=%s\n' "$status"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$AGENT_RUN/gpu_after_train.csv"
tail -n 80 "$AGENT_RUN/train_smoke_v1.log"
cat "$AGENT_RUN/gpu_after_train.csv"
# 保存这次 attempt 的 checkpoint 快照；不是“实际已更新到 step 2”的证明。
if [ -d .art ]; then
  tar -czf "$AGENT_RUN/art_after_train_attempt.tar.gz" .art
fi
test "$status" -eq 0 || { echo '退出异常：保留证据，不继续评估/重训'; exit "$status"; }
grep -Fq '[lifecycle] Backend cleanup complete' "$AGENT_RUN/train_smoke_v1.log"
grep -Fq 'Training complete! Backend resources released.' "$AGENT_RUN/train_smoke_v1.log"
```

验收：从 step 0 开始、2 轮真实梯度更新而非持续 SKIP、清理完成标记、单元格自行结束、`train_exit=0`，且本次后端的 Python/vLLM 进程不再占用 GPU。`timeout` 是 20 分钟的安全上限，不是接受条件；124/137 等非零退出码不能算成功。不要因为出现完成文案就忽略退出码。Drive 的 `.art` 压缩包不推 Git。

**先人工比较 GPU 前后 CSV，确认没有本次后端遗留再继续。** 无关内核的 GPU 占用不要求归零；也不要全局 `pkill python`。

## 6. 用新 checkpoint 做单场景评估，检查 JSON、step 与退出

默认仅 1 条 deterministic rollout，避免为生命周期检查重复全量 450-rollout 审计。评估不训练、不改变权重。

```bash
%%bash
set -euo pipefail
cd "$AGENT_SMOKE"
"$AGENT_PY" - <<'PY'
from pathlib import Path
lines = Path('data/scenarios_val.jsonl').read_text().splitlines()
Path('/tmp/agenticrl_lifecycle_scenario.jsonl').write_text(lines[0] + '\n')
PY
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$AGENT_RUN/gpu_before_eval.csv"
set +e
timeout --signal=INT --kill-after=30s 20m \
  "$AGENT_PY" -u scripts/eval_checkpoint.py \
    --config configs/train_config.yaml \
    --scenarios /tmp/agenticrl_lifecycle_scenario.jsonl \
    --mode deterministic --batch-scenarios 1 --reward-version v1 \
    --complete-chain --compare-rewards --save-traces \
    --output "$AGENT_RUN/eval_exit_check.json" \
    > "$AGENT_RUN/eval_exit_check.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$AGENT_RUN/eval_exit_code.txt"
printf 'eval_exit=%s\n' "$status"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$AGENT_RUN/gpu_after_eval.csv"
tail -n 60 "$AGENT_RUN/eval_exit_check.log"
cat "$AGENT_RUN/gpu_after_eval.csv"
test "$status" -eq 0 || { echo '评估退出异常：停止'; exit "$status"; }
grep -Fq '[lifecycle] Backend cleanup complete' "$AGENT_RUN/eval_exit_check.log"
grep -Fq 'Evaluation complete! Backend resources released.' "$AGENT_RUN/eval_exit_check.log"
"$AGENT_PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['AGENT_RUN'])
r = json.loads((p / 'eval_exit_check.json').read_text())
assert r['step'] == 2, f"实际 checkpoint step={r['step']}，不能宣称两步训练完成"
assert r['total_scenarios'] == 1 and r['total_rollouts'] == 1
assert not r['has_infra_failures'], '推理基础设施失败，先排查日志'
print('JSON/step/退出检查通过；还需人工核对 GPU 前后 CSV')
PY
```

此一场景检查没有统计意义，不据此报告模型能力提升。完整验收仍需单元格自动结束、退出码 0、JSON 完整与本次后端 GPU 进程消失全部成立。

## 7. 可选：退出检查通过后再跑完整 V1 审计

只有你需要重新观察这个**新 smoke checkpoint**的质量时才执行。不是修复验收的必需工作。

```bash
%%bash
set -euo pipefail
cd "$AGENT_SMOKE"
set +e
timeout --signal=INT --kill-after=30s 30m \
  "$AGENT_PY" -u scripts/eval_checkpoint.py \
    --config configs/train_config.yaml \
    --scenarios data/scenarios_val.jsonl \
    --mode both --generations 8 --batch-scenarios 4 --reward-version v1 \
    --complete-chain --compare-rewards --save-traces \
    --output "$AGENT_RUN/eval_step2_reward_audit.json" \
    > "$AGENT_RUN/eval_step2_reward_audit.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$AGENT_RUN/eval_audit_exit_code.txt"
printf 'audit_exit=%s\n' "$status"
tail -n 80 "$AGENT_RUN/eval_step2_reward_audit.log"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$AGENT_RUN/gpu_after_audit.csv"
exit "$status"
```

全量为 50 条 deterministic + 400 条 sample。不同 checkpoint 或不同环境不能直接归因比较学习增益。超时/非零退出同样是未通过。

## 8. 返回哪些文件

优先返回或按已有 SSH feedback 分支流程推送以下小文件：

- `source_commit.txt`、`checkpoint_source.txt`、`input_sha256.txt`
- `packages.txt`、`dependency_check.log`、`backend_preflight.log`、`pytest.log`
- `train_smoke_v1.log`、`train_exit_code.txt`、`gpu_before_train.csv`、`gpu_after_train.csv`
- `eval_exit_check.log`、`eval_exit_check.json`、`eval_exit_code.txt`、GPU 前后 CSV
- 若安装失败，再提供 `install.log` / `peft_fix_install.log`
- 若执行了可选审计，再提供对应日志、JSON、退出码和 GPU CSV

不用上传 `.art` 权重包来分析退出问题；权重备份保留在 Drive。反馈收集器如有旧白名单，要明确补入这次的退出码、CSV 与环境记录，不能遗漏或补造成功记录。

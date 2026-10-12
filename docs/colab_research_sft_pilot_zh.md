# Colab A100：最小 SFT 更新 / Transformers 重载 pilot

本轮只新增本地入口与合同测试；**尚无 A100 实测结果**。本指南不启动正式 SFT、GRPO、validation 选点或 test。远程分支：`research/sft-pilot-v1-20261012`；实际 commit 请核对助手最后回复。

固定合同：官方 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`；冻结 train 中按文件顺序各取第一条深度 1/2/3，总计 3 条；batch=3，两次 LoRA 更新，r=8/alpha=16/dropout=0、七类 projection、lr=1e-4、bf16/eager、seed=20261011，无 packing/截断/重试/续训。完整有效配置见 `configs/research/sft_pilot_v1.json`。这是兼容诊断，不是有效学习实验。

成功条件：两次更新分别有有限正 loss/grad 与权重变化；HF loss 与独立 shifted assistant-only CE 一致；保存 safetensors 与 Base/step 绑定；**新建 Base + PEFT** 重载，adapter tensor hash 完全一致，固定 train prefix 的 logits 在预设 bf16 容差内一致。不是同一个模型对象自检，不以答案准确率挑选产物。

**本入口不验证 ART/vLLM。** 使用独立 Transformers/PEFT 环境，避免把上游 ART/Unsloth 与另一个 Transformers 栈混装。完成后只可确认 SFT + Transformers 保存/重载；完整 R3.4 的 ART/vLLM、R3.5 正式配置/选点、R3.6 共享起点仍待后续授权及证据。

## 1. 服务器拉取（SSH；不要覆盖旧 `.art`）

全新目录推荐：

```bash
git clone --branch research/sft-pilot-v1-20261012 --single-branch \
  git@github.com:WafoeX/AgenticRL.git AgenticRL-sft-pilot-v1
cd AgenticRL-sft-pilot-v1
git rev-parse HEAD
git status --short
```

已有同分支的干净 checkout 才更新（不在旧 master 工作目录直接切换训练身份）：

```bash
cd AgenticRL-sft-pilot-v1
test -z "$(git status --porcelain)" || { echo '先保存本地改动'; exit 1; }
test "$(git branch --show-current)" = research/sft-pilot-v1-20261012
git pull --ff-only origin research/sft-pilot-v1-20261012
```

SSH agent 必须能被当前 shell 使用；首次连接核对 GitHub 主机指纹，不关闭 host-key 检查，不提交密钥/token。不能连接时先停，不改成带 Token 的 URL。本次本地22端口超时，已确认443可访问；服务器也遇到22被封时，可先执行以下环境设置再用原clone/pull/push命令（仍需正确known_hosts）：

```bash
export GIT_SSH_COMMAND='ssh -o ConnectTimeout=15 -o HostName=ssh.github.com -p 443 -o HostKeyAlias=github.com'
```

Colab中若需跨单元格继承，在Python格使用：

```python
os.environ['GIT_SSH_COMMAND'] = 'ssh -o ConnectTimeout=15 -o HostName=ssh.github.com -p 443 -o HostKeyAlias=github.com'
```

## 2. Colab 设置路径与预算（Python 单元格）

选择 A100；在第一格记录**实际 GPU 分配时间**，不是训练开始时间。之前的安装、下载、空闲都计入 20 GPU 小时。不要把示例上限当剩余额度。先前消耗须从总额扣除；无法核对余额时停止。

```python
import os, sys, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive

assert sys.version_info >= (3, 11)
subprocess.run(['nvidia-smi'], check=True)
remaining = float(input('这次 GPU 分配时的实际剩余 GPU 小时（已扣除历史消耗）：'))
allocated = input('实际 GPU 分配 UTC 时间，例 YYYY-MM-DDTHH:MM:SSZ：').strip()
started = datetime.fromisoformat(allocated.replace('Z', '+00:00'))
assert started.tzinfo is not None
assert 0 < remaining <= 20
assert 0 <= (datetime.now(timezone.utc) - started).total_seconds() < remaining * 3600

drive.mount('/content/drive')
repo = Path('/content/AgenticRL-sft-pilot-v1')
run = Path('/content/drive/MyDrive/AgenticRL-research-sft-pilot') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run.mkdir(parents=True, exist_ok=False)
assert shutil.disk_usage('/content').free >= 15 * 1024**3, '本地空间不足，先停'
assert shutil.disk_usage(run).free >= 4 * 1024**3, 'Drive 可用空间不足，先停'
os.environ.update(SFT_REPO=str(repo), SFT_RUN=str(run),
                  SFT_PY='/content/sft-pilot-venv/bin/python',
                  SFT_REMAINING_HOURS=str(remaining), SFT_GPU_ALLOCATED_AT=allocated)
(run / 'allocation.json').write_text(__import__('json').dumps({
    'remaining_hours_at_allocation': remaining, 'allocated_at': allocated,
    'first_recorded_at': datetime.now(timezone.utc).isoformat()}, indent=2))
print('本次结果目录：', run)
```

重启 kernel 会丢失 env；恢复时沿用同一结果路径/分配时间，不创建新 run 掩盖成本。Drive 不存私钥。脚本不负责关掉计费 runtime；结束后由你主动断开/删除 runtime。

## 3. Colab 拉取、隔离安装（分别粘贴单元格）

`%%bash` 必须是第一行。仓库已存在时选第 1 节的更新流程，不删除旧目录。

```bash
%%bash
set -euo pipefail
test ! -e "$SFT_REPO"
git clone --branch research/sft-pilot-v1-20261012 --single-branch \
  git@github.com:WafoeX/AgenticRL.git "$SFT_REPO"
cd "$SFT_REPO"
git rev-parse HEAD | tee "$SFT_RUN/source_commit.txt"
git status --short | tee "$SFT_RUN/source_status.txt"
test -z "$(git status --porcelain)"
nvidia-smi > "$SFT_RUN/nvidia_before_install.txt"
```

不要安装旧 `requirements.txt` 的 ART backend 栈，不运行旧 notebook 的 50-step/清 `.art` 单元格。

```python
%pip install -q uv
```

```bash
%%bash
set -euo pipefail
cd "$SFT_REPO"
test ! -e /content/sft-pilot-venv
uv venv --python 3.12 /content/sft-pilot-venv
uv pip install --python "$SFT_PY" -r requirements-sft-pilot.txt \
  2>&1 | tee "$SFT_RUN/install.log"
uv pip check --python "$SFT_PY" | tee "$SFT_RUN/pip_check.log"
uv pip freeze --python "$SFT_PY" > "$SFT_RUN/packages.txt"
```

运行时 CUDA/driver 不兼容、依赖安装/空间检查失败时停止，回传日志；不要无序升降级重跑，也不要安装第二个 backend 覆盖此 venv。安装时间也计 GPU 分配时间。

## 4. 下载固定 Base，执行本地合同与只读 preflight

```bash
%%bash
set -euo pipefail
cd "$SFT_REPO"
"$SFT_PY" - <<'PY' > "$SFT_RUN/base_path.txt"
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',
      revision='7ae557604adf67be50417f59c2c2f167def9a775',
      allow_patterns=['*.json', '*.safetensors', 'merges.txt']))
PY
BASE="$(cat "$SFT_RUN/base_path.txt")"
export RESEARCH_PILOT_PYTHON="$SFT_PY"
export RESEARCH_TOKENIZER_DIR="$BASE"
CUDA_VISIBLE_DEVICES="" "$SFT_PY" -m pytest tests/ -q \
  2>&1 | tee "$SFT_RUN/pytest.log"
"$SFT_PY" scripts/freeze_research.py --check | tee "$SFT_RUN/freeze_check.log"
CUDA_VISIBLE_DEVICES="" "$SFT_PY" scripts/pilot_research_sft.py \
  --base-dir "$BASE" --output "$SFT_RUN/preflight" \
  2>&1 | tee "$SFT_RUN/preflight.log"
```

测试中 tokenizer/torch 子进程路径显式指定，路径错误是 fail，不 silently skip。合同测试只用 CPU 随机微型 Qwen/张量；不加载官方权重训练、不占 CUDA。preflight 校验 Base 文件身份、冻结纪律、3 条专家与 mask，仅加载 tokenizer、不加载模型权重。历史冻结 manifest 的 candidate 字样与原 commit 保留不改；额外 freeze/lock 才是冻结发布登记。

确认所有测试通过、无 unexpected skip，preflight 的 `depths=[1,2,3]`、正监督 token count、`training_performed=false`。不满足则不进入下一格。脚本固定拒绝混合 split、改变两次更新配置、输出目录覆盖以及脏 checkout 的 GPU 运行。

## 5. 单独确认后才启动 GPU pilot

**先查看以上日志，再运行这个 Python 授权格。** 只有你输入确认串后，下一格才可进入 GPU 更新；这不是正式训练授权。

```python
answer = input('仅授权3条train、batch3、2次LoRA更新+Transformers重载；不含ART/vLLM或正式训练。输入 RUN_TWO_UPDATE_SFT_PILOT：')
assert answer == 'RUN_TWO_UPDATE_SFT_PILOT', '未授权，停止'
os.environ['SFT_GPU_AUTHORIZED'] = answer
```

```bash
%%bash
set -euo pipefail
cd "$SFT_REPO"
test "${SFT_GPU_AUTHORIZED:-}" = RUN_TWO_UPDATE_SFT_PILOT
BASE="$(cat "$SFT_RUN/base_path.txt")"
test -z "$(git status --porcelain)"
# 一个新 run，不补采样、不自动重试。超时600s，TERM后最多30s强制退出。
set +e
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 600s \
  "$SFT_PY" -u scripts/pilot_research_sft.py \
  --base-dir "$BASE" --output "$SFT_RUN/pilot" \
  --run-gpu --confirm-pilot-only \
  --remaining-gpu-hours "$SFT_REMAINING_HOURS" \
  --gpu-allocated-at "$SFT_GPU_ALLOCATED_AT" \
  2>&1 | tee "$SFT_RUN/pilot.log"
statuses=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${statuses[0]}" > "$SFT_RUN/pilot_exit_code.txt"
printf '%s\n' "${statuses[1]}" > "$SFT_RUN/tee_exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SFT_RUN/after_pilot_utc.txt"
nvidia-smi > "$SFT_RUN/nvidia_after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$SFT_RUN/gpu_processes_after.csv"
test "${statuses[0]}" = 0 && test "${statuses[1]}" = 0
"$SFT_PY" - <<'PY'
import json, os
from pathlib import Path
x = json.loads((Path(os.environ['SFT_RUN'])/'pilot/result.json').read_text())
assert x['status'] == 'SFT-and-Transformers-reload-passed-ART-vLLM-pending', x
assert x['actual_updates'] == 2 and x['max_parameter_delta'] > 0
assert x['final_adapter_sha256'] == x['reloaded_adapter_sha256']
assert x['ART_vLLM_verified'] is False and x['formal_training'] is False
print('SFT/Transformers pilot passed; ART/vLLM still pending')
PY
```

timeout exit=124/137、OOM、保存/重载失败、预算不足，均保留原 run，停下回传；不手动再跑同目录、不删除 checkpoint、不把部分更新写成通过。被 watchdog 强制终止时 result.json 可能只含最后进度，**以外层退出码为准**。PID 查询若还有本 pilot 进程则不能声称清理通过；不要杀掉不属于此 run 的进程。

默认上限 600s，加清理预留 60s；入口再次核对从分配时间起的安装/准备消耗是否仍容许该上限。最终仍需记录 GPU 断开时间，将日志时段到真正释放期间的空闲纳入账本。

## 6. 回传轻量反馈，不上传权重

权重留 `$SFT_RUN/pilot/checkpoint-2/` 的 Drive 原件。回传：source commit/status、allocation、packages、pytest/freeze/preflight日志、install/pip_check、pilot.log/退出码、result/preflight、updates.jsonl、checkpoint_identity.json、nvidia/PID 查询。`selected_experts.json` 为3条 train，允许反馈；不附 test 或 Hugging Face cache。

可在干净 checkout 开独立反馈分支：

```bash
%%bash
set -euo pipefail
cd "$SFT_REPO"
RUN_ID="$(basename "$SFT_RUN")"
FEEDBACK="feedback/sft-pilot-$RUN_ID"
test ! -e "$FEEDBACK"
mkdir -p "$FEEDBACK"
# 仅这些轻量文件；不递归复制整个run/Drive/.art/venv。
for file in source_commit.txt source_status.txt allocation.json packages.txt \
  pytest.log freeze_check.log preflight.log install.log pip_check.log pilot.log \
  pilot_exit_code.txt tee_exit_code.txt after_pilot_utc.txt nvidia_before_install.txt \
  nvidia_after.txt gpu_processes_after.csv; do
  test ! -f "$SFT_RUN/$file" || cp "$SFT_RUN/$file" "$FEEDBACK/$file"
done
for file in result.json preflight.json updates.jsonl checkpoint_identity.json selected_experts.json; do
  test ! -f "$SFT_RUN/pilot/$file" || cp "$SFT_RUN/pilot/$file" "$FEEDBACK/$file"
done
# preflight-only失败时也保留已有receipt
if test -f "$SFT_RUN/preflight/result.json"; then
  cp "$SFT_RUN/preflight/result.json" "$FEEDBACK/preflight_result.json"
fi
"$SFT_PY" - <<'PY' > "$FEEDBACK/checkpoint_location.json"
import os, json
from pathlib import Path
p = Path(os.environ['SFT_RUN'])/'pilot/checkpoint-2'
print(json.dumps({'Drive_checkpoint_path': str(p), 'exists': p.is_dir()}, indent=2))
PY
git switch -c "feedback/sft-pilot-$RUN_ID"
git add "$FEEDBACK"
git diff --cached --stat
# 人工核对日志没有secret，再运行以下两行；无需上传safetensors。
git commit -m "feedback: bounded SFT pilot $RUN_ID"
git push -u origin "feedback/sft-pilot-$RUN_ID"
```

失败也回传，注明是否只完成 preflight。不要自动合并反馈分支，不把成功 reload 当能力提升。验收后再决定单独的 ART/vLLM pilot；**本指南没有这一步的自动安装/启动命令**。

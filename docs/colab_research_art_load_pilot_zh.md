# 新 Colab 运行时：从 Drive 恢复 SFT step 2，验证 ART/vLLM 加载

> 不必重做 SFT。下面只复用已经验收的 `checkpoint-2`，不更新权重、不启动 GRPO、不使用 test。
> 本地只做了语法、Ruff、Pyright/LSP 检查，**未运行 pytest，也未替你执行 GPU pilot**。ART/vLLM 的实际结果待这次远端反馈。

## 本次范围与判定

- 新 A100 40GB、独立 Python 3.12 ART 环境；不用旧 SFT venv。
- 从固定官方 revision 恢复 Base；校验原 Drive adapter 的三份文件 SHA 和完整身份。
- 将校验过的文件复制到全新本地 ART workspace 的 `checkpoints/0002`；只请求 `sft-load-pilot@2`，不请求未带 step 的名称。
- ART/Unsloth 实际加载的 LoRA 张量须与保存值一致；若发生 dtype 转换，逐张量记录并要求转换后精确一致。
- vLLM `/v1/models` 的 adapter root/parent 须正确；固定 train 专家前缀对 Base/adapter 各请求 **1 token**，记录共有 top-20 logprob 差异，确认不是静默回落到 Base。不是性能比较，也不宣称 Transformers/vLLM 全向量数值等价。
- 另外：一条明确标记为 **scripted/scaffolded** 的 validation observation 回传请求（最多 128 tokens），以及 validation 各深度首条的 **3 次 greedy 独立尝试**。不会跑 sample-8/pass@8。
- 最多 21 个生成请求：2 个单 token 对照 + 1 个 scaffolded 请求 + 3×6 轮模型生成；脚本和外层均设 900 秒进程上限。安装/下载及整个 Colab 会话费用**不在该进程计时内**；不要求手填剩余小时或 UTC 分配时间。
- 保留所有 raw 输出、错误与未返回槽位，不自动重跑。模型答错/工具参数错不伪装成基础设施失败。
- `ART_vLLM_load_verified=true` 只表示本次加载/请求关卡通过。原生工具调用是否实际覆盖，看 `generated_tool_parser_coverage`；GPU 释放还需看结束快照。**不自动宣布完整 R3.4、泛化收益或正式 B1 验收。**

ART 的 `register` 内部可能初始化 trainer/optimizer 对象；本入口不调用训练、optimizer step、保存新 checkpoint 或删除 checkpoint 的接口。

下面各框分别粘贴到 Colab **不同单元格，按顺序执行一次**。失败后先保留日志，不重复创建 run、不自行改依赖或合并权重。

## 第 1 格：选 A100、挂载 Drive、自动创建新记录目录（Python）

在 Colab“更改运行时类型”选择 A100。然后执行：

```python
from google.colab import drive
from datetime import datetime, timezone
from pathlib import Path
import os, shlex, shutil, subprocess

drive.mount('/content/drive')
env_file = Path('/content/agenticrl-art-verify.env')
if env_file.exists():
    raise RuntimeError('本运行时已经创建过 run；不要重复第 1 格。后续格直接读取现有 env 文件。')
if shutil.disk_usage('/content').free < 25 * 1024**3:
    raise RuntimeError('可用磁盘不足 25 GiB，先停止；不要删 Drive checkpoint。')
subprocess.run(['nvidia-smi'], check=True)
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
settings = {
    'AR_REPO': '/content/AgenticRL-art-verify',
    'AR_BRANCH': 'research/sft-pilot-v1-20261012',
    'AR_ENV': '/content/agenticrl-art-verify-venv',
    'AR_PY': '/content/agenticrl-art-verify-venv/bin/python',
    'AR_SRC': '/content/drive/MyDrive/AgenticRL-research-sft-pilot/20261012T080000Z/pilot/checkpoint-2',
    'AR_WORK': '/content/agenticrl-art-verify-work/' + run_id,
    'AR_RUN': '/content/drive/MyDrive/AgenticRL-art-load-pilot/' + run_id,
}
for f in ('adapter_config.json', 'adapter_model.safetensors', 'research_checkpoint.json'):
    if not (Path(settings['AR_SRC']) / f).is_file():
        raise FileNotFoundError(str(Path(settings['AR_SRC']) / f))
Path(settings['AR_RUN']).mkdir(parents=True, exist_ok=False)
os.environ.update(settings)
env_file.write_text(''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k,v in settings.items()))
Path(settings['AR_RUN'], 'session_note.json').write_text(__import__('json').dumps({
    'recording_started_at': datetime.now(timezone.utc).isoformat(),
    'actual_GPU_allocation_started_at': None,
    'historical_GPU_hours': None,
    'total_GPU_budget_verified': False,
    'training_authorized': False,
}, indent=2) + '\n')
print('原件（不改）：', settings['AR_SRC'])
print('这次反馈目录：', settings['AR_RUN'])
```

不是 A100 或有文件缺失时先停止。新运行时的旧 SSH key 也可能已消失：下面公开仓库使用**匿名 HTTPS 只读 clone，不需要 Token**；若仓库要求认证，请先恢复自己的 SSH 配置，改用 `git@github.com:WafoeX/AgenticRL.git`，不要把密钥/Token 写入 notebook 或反馈。

## 第 2 格：拉取入口，安装独立 ART 环境（`%%bash`）

不安装 `requirements-sft-pilot.txt`，不运行任何 pytest。让 ART backend 自身解析配套 vLLM/Torch/Unsloth；实际版本全部记录，不能把它说成旧 SFT 的 torch 2.8/Transformers 4.57.1 环境。

```bash
%%bash
set -euo pipefail
source /content/agenticrl-art-verify.env
if test -d "$AR_REPO/.git"; then
  test -z "$(git -C "$AR_REPO" status --porcelain)"
  test "$(git -C "$AR_REPO" branch --show-current)" = "$AR_BRANCH"
  git -C "$AR_REPO" pull --ff-only origin "$AR_BRANCH"
else
  git clone --single-branch --branch "$AR_BRANCH" \
    https://github.com/WafoeX/AgenticRL.git "$AR_REPO"
fi
test -f "$AR_REPO/scripts/pilot_research_art.py"
git -C "$AR_REPO" rev-parse HEAD | tee "$AR_RUN/source_commit.txt"
python -m pip install -q uv
uv --version | tee "$AR_RUN/uv_version.txt"
test ! -e "$AR_ENV"
uv venv --python 3.12 --seed "$AR_ENV" 2>&1 | tee "$AR_RUN/venv.log"
uv pip install --python "$AR_PY" --index-url https://pypi.org/simple \
  'openpipe-art[backend]==0.5.17' 'peft==0.18.0' 'torchao==0.15.0' \
  'psutil==7.0.0' 'PyYAML==6.0.2' \
  2>&1 | tee "$AR_RUN/install.log"
"$AR_PY" -m pip check 2>&1 | tee "$AR_RUN/pip_check.txt"
"$AR_PY" -m pip freeze --all > "$AR_RUN/pip_freeze.txt"
"$AR_PY" --version > "$AR_RUN/python_version.txt"
printf 'installation completed\n' > "$AR_RUN/dependencies_ready.txt"
```

若 pip/uv 解析、下载或 `pip check` 失败，不继续第 3–5 格；先发 `install.log`。不在 Colab kernel 混装，也不自动降级/换版本重试。

## 第 3 格：仅恢复固定 Base 文件，不重新训练（`%%bash`）

Base 下载到与原 PEFT 配置一致的 Colab `/root/.cache/huggingface/hub`。不修改 `adapter_config.json`，也不改其 base 路径；不从最新 main 加载。

```bash
%%bash
set -euo pipefail
source /content/agenticrl-art-verify.env
test -f "$AR_RUN/dependencies_ready.txt"
"$AR_PY" - <<'PY' 2>&1 | tee "$AR_RUN/base_restore.log"
import os
from pathlib import Path
from huggingface_hub import snapshot_download
base = snapshot_download(
    'Qwen/Qwen2.5-0.5B-Instruct',
    revision='7ae557604adf67be50417f59c2c2f167def9a775',
    cache_dir='/root/.cache/huggingface/hub',
    allow_patterns=['*.json', '*.safetensors', 'merges.txt'],
)
Path(os.environ['AR_RUN'], 'base_path.txt').write_text(base + '\n')
print(base)
PY
```

## 第 4 格：先做无模型加载的身份/data preflight（`%%bash`）

```bash
%%bash
set -euo pipefail
source /content/agenticrl-art-verify.env
AR_BASE="$(< "$AR_RUN/base_path.txt")"
cd "$AR_REPO"
"$AR_PY" scripts/pilot_research_art.py \
  --base-dir "$AR_BASE" --adapter-dir "$AR_SRC" \
  --output "$AR_WORK/preflight" 2>&1 | tee "$AR_RUN/preflight.log"
printf 'preflight passed\n' > "$AR_RUN/preflight_ready.txt"
```

须看到 `status=preflight-passed-no-model-load`、`GPU_requested=false`。这里只规划 3 个槽位，`returned_attempts=0` 是预期的，不是 GPU 实验失败。校验不通过时不改 JSON/hash、不中途合并权重。

## 第 5 格：显式执行 ART/vLLM 加载与少量推理（`%%bash`）

只读原件、全新本地工作区、固定 step 2；关闭外部 W&B，恢复完成后禁止 HF 自动下载其他版本。只有这一格运行模型。

```bash
%%bash
set -euo pipefail
source /content/agenticrl-art-verify.env
test -f "$AR_RUN/preflight_ready.txt"
AR_BASE="$(< "$AR_RUN/base_path.txt")"
cd "$AR_REPO"
unset WANDB_API_KEY OPENAI_API_KEY
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi > "$AR_RUN/nvidia_before.txt"
set +e
timeout --signal=INT --kill-after=30s 900s \
  "$AR_PY" -u scripts/pilot_research_art.py \
  --base-dir "$AR_BASE" --adapter-dir "$AR_SRC" \
  --output "$AR_WORK/pilot" --run-art --confirm-load-only \
  2>&1 | tee "$AR_RUN/pilot.log"
CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${CODES[0]}" > "$AR_RUN/pilot_exit_code.txt"
printf '%s\n' "${CODES[1]}" > "$AR_RUN/tee_exit_code.txt"
nvidia-smi > "$AR_RUN/nvidia_after.txt" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
  > "$AR_RUN/gpu_processes_after.csv" 2>&1
set -e
printf 'pilot exit=%s; tee exit=%s\n' "${CODES[0]}" "${CODES[1]}"
# 即使失败也先执行第 6 格保存证据，不重复跑本格。
```

900 秒超时会发送 SIGINT，给现有生命周期清理留 30 秒，仍不结束才强制终止该进程组；**不能仅凭 exit code 声称 GPU 已清理**。若结束快照还有计算进程，先保留 PID/日志并停止该 Colab 运行时；不要 `pkill` 全机进程。

## 第 6 格：保存轻量反馈到 Drive（Python）

成功或失败都执行。只复制 JSON/JSONL/log，不重复上传 Base/adapter，不复制 venv、整个 ART workspace 或任何密钥。

```python
from pathlib import Path
import os, shutil
work = Path(os.environ['AR_WORK'])
run = Path(os.environ['AR_RUN'])
for phase in ('preflight', 'pilot'):
    source = work / phase
    if not source.exists():
        continue
    target = run / phase
    target.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
            shutil.copy2(path, target / path.name)
    for path in source.rglob('*.log'):
        if path.parent == source:
            continue
        dest = run / (phase + '_backend_logs') / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
print('已保存：', run)
result = run / 'pilot/result.json'
if result.is_file():
    import json
    obj = json.loads(result.read_text())
    print({k: obj.get(k) for k in ('status','ART_vLLM_load_verified',
          'planned_attempts','returned_attempts','missing_attempts','infra_failures',
          'answer_correct','generated_tool_parser_coverage','wall_seconds','error')})
zip_path = shutil.make_archive(str(run.parent / run.name), 'zip', root_dir=run)
print('轻量反馈 ZIP：', zip_path)
```

把 **反馈 ZIP** 或整个反馈目录内容发回来即可；也可以按你已有 SSH 流程推到新的 `feedback/art-load-<runid>` 分支，别直接推 master。不要发送 SSH key/HF Token，也不需要再上传 17 MB adapter。

最后检查：退出码均 0、`planned=returned=3`、`missing=infra=0`、adapter/Base probe 差异大于 1e-4、实际 LoRA 身份验算通过；完整工具解析覆盖与结束 GPU 进程快照需分别复核。`answer_correct` 只是 3 条开发诊断结果，scaffolded 前缀不算模型成功。

**资料复制完后主动断开/删除 Colab 运行时，防止 GPU 闲置计费。** Drive 的原 `checkpoint-2` 和这次反馈会保留，不自动进入正式 SFT/GRPO。

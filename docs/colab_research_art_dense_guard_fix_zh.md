# ART pilot：修复 Unsloth 8-bit 标记误判

这次 c691f5d 已越过 PEFT/torchao 版本错误，vLLM 显示 Base 从固定 snapshot 以 bfloat16、quantization=None 加载。但入口在验算 LoRA 前，误把 Unsloth 的 `is_loaded_in_8bit=True` 当作实际量化，3 个生成槽位仍全部缺失。没有训练或模型成功率证据。

已核对 Unsloth 2026.3.3 官方 wheel 的 `models/llama.py`：它对内部模型无条件设置 `is_loaded_in_8bit=True`，注释明确为禁用错误 DDP 的兼容处理。因此不能靠这个单一标记判定存储精度；也不能只看 vLLM 的日志，推断独立 ART/Unsloth 模型没有量化。

新检查记录该标记，但改用实际证据：4-bit 标记/量化配置和实际 ART init/engine args，所有参数的普通 Parameter/Tensor 类型及浮点 dtype，全部 24×7 个基础 projection 必须为普通 torch.nn.Linear 且 BF16；随后仍逐张量验算 LoRA 与保存文件的一致性（显式 dtype 转换必须记录）。没有改权重、依赖或 installed library，没有关闭量化验收。记录输出为 `loader_quantization_audit.json`。

一并排除自动生成的 `/unsloth_compiled_cache/`，避免缓存导致 clean-checkout 检查误退出。原源码仍须干净。仅语法/Ruff/Pyright/LSP 检查，无 pytest、本地 GPU 或模型推理；后续实际服务兼容仍待反馈。

## 操作

先用原指南第 6 格保存当前失败反馈（已保存则不重复）。NCCL `destroy_process_group` 警告、torchao C++ 扩展不兼容警告和导入顺序警告均要保留；不能说零告警或已优雅释放。

### 第 1 格：确认 GPU 没有遗留计算进程，再拉取修复

```bash
%%bash
set -euo pipefail
source /content/agenticrl-art-verify.env
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
  | tee "$AR_RUN/dense_guard_gpu_before_retry.csv"
if test -s "$AR_RUN/dense_guard_gpu_before_retry.csv"; then
  echo '停止：还有 GPU 计算进程。请先发送上面 PID 快照，不要直接重跑或杀全机进程。'
  exit 1
fi
cd "$AR_REPO"
if test -n "$(git status --porcelain)"; then
  git status --short
  echo '停止：源码工作区有改动，先核对，不自动 reset。'
  exit 1
fi
test "$(git branch --show-current)" = "$AR_BRANCH"
git pull --ff-only origin "$AR_BRANCH"
"$AR_PY" -m pip check
```

若出现遗留进程或源码改动，停下发送输出；不自动升级依赖、清空缓存、删除结果或启动模型。

### 第 2 格：创建新 run，保留旧目录和 Base 文件（Python）

```python
from datetime import datetime, timezone
from pathlib import Path
import os, shlex, shutil, json, subprocess
old_run = Path(os.environ['AR_RUN'])
assert (old_run / 'base_path.txt').is_file()
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-denseguard'
new_run = old_run.parent / run_id
new_work = Path('/content/agenticrl-art-verify-work') / run_id
assert not new_work.exists()
new_run.mkdir(parents=True, exist_ok=False)
for name in ('base_path.txt', 'python_version.txt', 'uv_version.txt'):
    if (old_run / name).is_file():
        shutil.copy2(old_run / name, new_run / name)
py = os.environ['AR_PY']
(new_run / 'pip_freeze.txt').write_text(subprocess.check_output([py, '-m', 'pip', 'freeze', '--all'], text=True))
(new_run / 'pip_check.txt').write_text(subprocess.check_output([py, '-m', 'pip', 'check'], text=True))
(new_run / 'source_commit.txt').write_text(subprocess.check_output(['git','-C',os.environ['AR_REPO'],'rev-parse','HEAD'], text=True))
(new_run / 'dependencies_ready.txt').write_text('existing pinned ART environment retained\n')
(new_run / 'session_note.json').write_text(json.dumps({
    'previous_failed_run': str(old_run),
    'repair': 'physical dense-tensor audit instead of Unsloth 8bit DDP marker',
    'recording_started_at': datetime.now(timezone.utc).isoformat(),
    'actual_GPU_allocation_started_at': None,
    'total_GPU_budget_verified': False,
    'training_authorized': False,
}, indent=2) + '\n')
os.environ['AR_RUN'] = str(new_run)
os.environ['AR_WORK'] = str(new_work)
keys = ('AR_REPO','AR_BRANCH','AR_ENV','AR_PY','AR_SRC','AR_WORK','AR_RUN')
Path('/content/agenticrl-art-verify.env').write_text(''.join(
    'export ' + k + '=' + shlex.quote(os.environ[k]) + '\n' for k in keys))
print('旧结果保留：', old_run)
print('新反馈目录：', new_run)
```

然后仅执行原指南 **第 4→5→6 格**。不用原第 1–3 格，不用重装依赖/下载 Base/重训。每次仍为固定三条 validation greedy 加有限控制 probe；保存反馈后主动断开运行时，避免闲置 GPU 计费。出现下一处失败就回传，不自动反复重试或降低验收条件。

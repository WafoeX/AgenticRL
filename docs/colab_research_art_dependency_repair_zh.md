# ART 加载失败：PEFT/torchao 窄依赖修复

## 已知事实

此次 `cea396c` 运行在模型注册/LoRA 注入阶段失败：ART 0.5.17 固定 torchao 0.15.0，而自动解析的 PEFT 0.21.2 要求 torchao >=0.16.0。3 个生成槽位全部未返回，未训练；不能把 `infra_failures=0` 解释成没有基础设施问题，也不能把缺失槽位当 0% 正确率。`tracking 0 owned processes` 不是 GPU 释放证明，仍检查结束快照。

已检查官方 wheel 的声明/源码：
- Unsloth 2026.3.3 和 Zoo 2026.3.1 要求 PEFT >=0.18.0；**不能使用旧 SFT 的 0.17.1**。
- PEFT 0.18.0 的 `is_torchao_available()` 下限是 0.4.0，接受 0.15.0。
- 因此只固定 **PEFT 0.18.0 / torchao 0.15.0**，不升级 ART 固定的 torchao，不变更 Torch/vLLM/Unsloth，不改 checkpoint。

没有本地运行 pytest 或 GPU。此修复消除已定位的依赖条件冲突，**不保证后续加载/服务环节全部通过**。

## 先保留旧结果

若原指南第 6 格尚未执行，先执行它保存失败证据；执行过就不重复。保存完成后按下面修复。不要删除旧 `pilot`、改旧 checkpoint JSON/hash，或重跑原第 1/2 格。

### 修复格 1：保留依赖前后记录，拉取新代码（`%%bash`）

```bash
%%bash
set -euo pipefail
source /content/agenticrl-art-verify.env
cd "$AR_REPO"
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = "$AR_BRANCH"
"$AR_PY" -m pip freeze --all > "$AR_RUN/repair_before_freeze.txt"
git pull --ff-only origin "$AR_BRANCH"
uv pip install --python "$AR_PY" --index-url https://pypi.org/simple \
  'peft==0.18.0' 'torchao==0.15.0' 2>&1 | tee "$AR_RUN/repair_install.log"
"$AR_PY" -m pip check 2>&1 | tee "$AR_RUN/repair_pip_check.txt"
"$AR_PY" -m pip freeze --all > "$AR_RUN/repair_after_freeze.txt"
"$AR_PY" - <<'PY'
from importlib.metadata import version
for package, expected in [('openpipe-art','0.5.17'), ('peft','0.18.0'), ('torchao','0.15.0')]:
    actual = version(package)
    print(package, actual)
    assert actual == expected, (package, actual, expected)
PY
nvidia-smi | tee "$AR_RUN/repair_gpu_snapshot.txt"
```

这格不导入模型、不训练、不运行测试。若安装或 `pip check` 失败，停在这里发送日志；不要用 `--no-deps` 强行忽略冲突。若有遗留 GPU 计算进程，先保存快照并停止运行时，不盲目杀全机进程。

### 修复格 2：新建重试目录，复用已下载 Base（Python）

不需要新运行时，不需要重新下载 Base，不覆盖旧结果。下面只修改 notebook 非密钥环境设置，原件仍指向同一 Drive checkpoint。

```python
from datetime import datetime, timezone
from pathlib import Path
import os, shlex, shutil, json
old_run = Path(os.environ['AR_RUN'])
base_path_file = old_run / 'base_path.txt'
assert base_path_file.is_file(), base_path_file
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-peft018'
new_run = old_run.parent / run_id
new_work = Path('/content/agenticrl-art-verify-work') / run_id
assert not new_work.exists(), new_work
new_run.mkdir(parents=True, exist_ok=False)
for name in ('base_path.txt', 'python_version.txt', 'uv_version.txt'):
    if (old_run / name).is_file():
        shutil.copy2(old_run / name, new_run / name)
shutil.copy2(old_run / 'repair_after_freeze.txt', new_run / 'pip_freeze.txt')
shutil.copy2(old_run / 'repair_pip_check.txt', new_run / 'pip_check.txt')
(new_run / 'dependencies_ready.txt').write_text('narrow PEFT repair completed\n')
(new_run / 'session_note.json').write_text(json.dumps({
    'previous_failed_run': str(old_run),
    'recording_started_at': datetime.now(timezone.utc).isoformat(),
    'actual_GPU_allocation_started_at': None,
    'total_GPU_budget_verified': False,
    'training_authorized': False,
    'repair': 'peft 0.21.2 -> 0.18.0; torchao remains 0.15.0',
}, indent=2) + '\n')
os.environ['AR_RUN'] = str(new_run)
os.environ['AR_WORK'] = str(new_work)
keys = ('AR_REPO','AR_BRANCH','AR_ENV','AR_PY','AR_SRC','AR_WORK','AR_RUN')
Path('/content/agenticrl-art-verify.env').write_text(''.join(
    'export ' + k + '=' + shlex.quote(os.environ[k]) + '\n' for k in keys))
print('旧结果保留：', old_run)
print('新反馈目录：', new_run)
```

接着只执行原指南 **第 4 → 第 5 → 第 6 格**（新的 preflight、新的一次显式加载、保存反馈）。不执行原第 1–3 格，不增加样本或更新次数。成功或失败都回传新的轻量反馈 ZIP；出错不自动继续换版本。资料保存后断开运行时，避免空闲计费。

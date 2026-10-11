# Colab：训练/评估完成后不退出、GPU 不释放的修复

## 结论与证据边界

已修复两个能够直接造成生命周期异常的代码缺陷：

1. `src/train.py` 和 `scripts/eval_checkpoint.py` 以前创建 `LocalBackend()` 后，没有在成功、注册失败或训练/评估异常时显式 `await backend.close()`。
2. 固定版本 ART 0.5.17 的 `mp_actors.Proxy.close()` 只对直属模型服务进程执行 `terminate/join/kill`，不递归清理其 vLLM 后代。`LocalBackend.close()` 对代理调用的 `close()` 也是代理自身的方法，不是远程 `UnslothService.close()`。ART 的 monitor 完成回调还会 `pop` 服务并关闭代理，因此不能等到 `asyncio.run()` 自动退出时再依赖其清理。

之前的 Colab 反馈中，评估已经打印 `Detailed results saved`，JSON 完整；主进程结束后仍有 PPID=1 的 Python/vLLM 进程，GPU util=0、约 37.6GB 显存被占用。这与上述后代进程遗留机制吻合。**保存模型/JSON成功，不等于进程和资源已退出。** 遗留子进程还可能持有输出管道，使 Colab 单元格或 `tee` 一直等待 EOF；这是可能的解释，不能凭现存日志证明当时具体等待点。

原 Colab 运行时已删除，无法补抓当时的线程栈、退出码。这里确认的是源代码缺陷和 CPU 可复现机制，不声称已经在用户原 A100 环境验证 GPU 释放，也不把异常退出改记为成功。

### 核对的上游版本

从 PyPI 下载 `openpipe-art==0.5.17` wheel，只提取源码，没有安装 GPU 后端。wheel SHA-256：

```text
97b552444adbefe95807e24199190517e29a068e281bac2299e0678936be9f63
```

检查了 wheel 中的：

- `art/local/backend.py`：`close()` / `_close()` / monitor done callback。
- `mp_actors/move.py`：`Proxy.close()` / atexit 清理。
- `art/unsloth/service.py`：共享模式的 in-process vLLM 与 dedicated 模式的 `_vllm_process`。

并用 wheel 原样的 **真实 `mp_actors` 代理**启动 CPU 父子进程：单独 `close_proxy()` 后，子进程仍活着；加入本次清理后，同样的子进程停止，探针退出码为 0。它不是 GPU 推理测试。

## 修复内容

新增 `src/backend_lifecycle.py`，两个入口使用统一 `managed_backend()`：

- 注册、训练/验证、评估、结果写盘均在清理作用域内；成功、异常和 Python 异步取消都会执行清理。
- **先记录后端所属进程树，再关闭 ART 代理**，否则父进程终止后，后代可能已重新挂到 PID 1，失去归属线索。
- 只识别 ART 0.5.17 服务对象的 `_process` / `_vllm_process` 句柄，且根进程必须是当前脚本的直属子进程。
- 用 PID + 创建时间保留身份，检查已退出的原始进程句柄，避免 PID 重用导致误杀。
- 保留原来的 monitor 异常容错，但统一安装且不重复套补丁；记录本后端的 monitor，在显式关闭后取消它们，不取消 Colab 内核或其他 asyncio 任务。
- 对已记录的残留进程先 TERM，等待最多 5 秒；仍存活则 KILL，再等待最多 2 秒。这两个超时是进程回收等待时间，不是整个 GPU 初始化或 ART `close()` 的总超时。
- 正常工作若清理失败，命令报错，不打印“资源已释放”；若本来已有异常，保留原异常并附加清理失败信息。
- 不删除 `.art`、checkpoint、模型、评估 JSON；训练原有的 checkpoint 保留策略不变。
- 新增直接依赖 `psutil>=5.9`，无需升级 ART、torch、vLLM 或 PEFT。

完成标记现在区分：

```text
Training updates complete! ... closing backend...
[lifecycle] Closing ART backend; tracking N owned process(es)
[lifecycle] Backend cleanup complete
Training complete! Backend resources released. ...
```

评估先保存 JSON，再清理，最后才输出：

```text
Detailed results saved to ...
[lifecycle] Closing ART backend; tracking N owned process(es)
[lifecycle] Backend cleanup complete
Evaluation complete! Backend resources released.
```

这些标记仍不能单独代替 shell 退出码和 GPU 进程核对。

## 本地验证

隔离 Python 3.13.15 环境，未安装或启动 GPU 后端：

- 全套 pytest：**173 passed**，原 150 项测试保留，新增 23 项生命周期回归。
- 回归覆盖：2-step mock 训练、单/双模式评估、注册/获取步数/工作异常、JSON 写入失败、清理失败不伪报成功、取消、monitor 补丁幂等、PID 重用保护。
- 四项真实进程树回归覆盖：TERM 正常退出 / 忽略 TERM 需 KILL，以及后端正常关闭 / 关闭报错。均验证无关兄弟进程继续运行、checkpoint 字节不变。
- wheel 原样 `mp_actors` 代理 CPU 探针：旧关闭路径留下子进程；新关闭路径回收子进程，退出码 0。
- 独立 Pyright 对新增 helper/test 无错误；入口剩余 ART 导入提示来自本地未安装 GPU 后端。曾出现新增模块的 LSP 导入缓存误报，已用新进程 Pyright 和实际 Python import 复核，不通过源码忽略注释掩盖。

## Colab 最小验证：不需要重训

**先通过 SSH 拉取包含本次修复的新代码，并记录实际 `git rev-parse HEAD`。** 全新 Colab 的拉取、依赖、2-step 训练与退出验收代码见 [Colab 生命周期验证命令](colab_lifecycle_validation_zh.md)。不逐格执行旧 notebook 的清空 `.art` / 50-step 训练流程。

### 1. 先在 Colab CPU 环境跑生命周期回归

按已有指南设置 `AGENT_REPO` / `AGENT_PY`，指向当前仓库与实际实验 Python。不要误用删除前的路径。只补小型依赖，不重新安装整套 GPU 栈：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
"$AGENT_PY" -m pip install 'psutil>=5.9' pytest pytest-asyncio pyyaml
"$AGENT_PY" -m pytest -q tests/test_backend_lifecycle.py
```

此测试启动的是临时 CPU 进程，不会使用 GPU，也不会访问实验 `.art`。预期 **23 passed**；Linux 的 zombie 进程不算仍持有 GPU 的活进程。

### 2. 若 Drive 有 checkpoint，只评估一个场景

在新的 Colab GPU 运行时按 `docs/colab_phase3_validation_zh.md` 恢复完整 `.art`、兼容环境和相同模型身份。**没有备份 checkpoint 时，不能把新运行声称为原 step 2；不要求为本次退出验证重训。** 可先只完成第 1 步，GPU 验收保持待定。

已有 checkpoint 和环境时，创建单场景临时输入；不改原数据，不重跑全量审计：

```bash
%%bash
set -euo pipefail
cd "$AGENT_REPO"
"$AGENT_PY" - <<'PY'
from pathlib import Path
lines = Path('data/scenarios_val.jsonl').read_text().splitlines()
Path('/tmp/agenticrl_exit_scenario.jsonl').write_text(lines[0] + '\n')
PY
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
set +e
"$AGENT_PY" -u scripts/eval_checkpoint.py \
  --config configs/train_config.yaml \
  --mode deterministic --generations 1 --batch-scenarios 1 \
  --scenarios /tmp/agenticrl_exit_scenario.jsonl \
  --output /tmp/agenticrl_exit_check.json \
  > /tmp/agenticrl_exit_check.log 2>&1
status=$?
set -e
printf 'eval_exit=%s\n' "$status"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
exit "$status"
```

如果原实验模型的 name/project/base 与配置不同，先提供相同身份的 YAML，再替换 `--config` 路径；不要修改原训练配置或误读其他 checkpoint。

验收要求：

1. 日志的 `Loaded checkpoint: step 2` 确认恢复的是原 checkpoint。
2. JSON 可解析、单场景结果已保存。
3. 出现 `Backend cleanup complete` 和最终完成标记。
4. 单元格自行结束，`eval_exit=0`。
5. `nvidia-smi` 对比前后，本次后端的 Python/vLLM 进程不再持有显存；不要求同一 GPU 上其他内核/用户的进程也消失。
6. 保留日志、JSON、实际代码提交与环境记录。若未退出，不要反复重新启动训练；记录进程树和显存占用，继续定位。

## 限制

- 本轮没有访问新 Colab，也没有声称已完成 GPU 退出验收。
- 适配器针对固定 ART 0.5.17 的私有进程句柄；升级 ART 后须重新检查上游生命周期和字段。
- 只回收当前后端能够证明归属的进程，不扫描/杀掉所有 Python、vLLM 或整块 GPU 的进程。
- 运行时被直接删除、SIGKILL 或 OS 强制终止时，Python 的 finally 无法执行。此次修复不能替代 checkpoint 的 Drive 备份。
- 使用 `python -m src.train` / `python scripts/eval_checkpoint.py` 入口才包含本次清理；绕过入口、在 notebook 中另写的直接 ART 调用不自动获益。
- 原先的 monitor 健康检查容错策略保留；本轮不重新设计健康检测、W&B、奖励或训练算法。

# RunPod 配置指南

这是在 RunPod 上运行本项目的经过验证的流程，请按阶段顺序执行。

**为什么选 RunPod 而不是 Colab：** ART 要求 torch >= 2.8.0，而 Colab 的默认 A100 运行时通常没有这个版本。在 Colab 中原地升级 torch 可能成功，但过程不稳定；RunPod 提供与 ART 精确匹配的 PyTorch 2.8 镜像（`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`），少一个容易出错的环节。

**总成本：** 约 4–6 美元（A100 80GB 约 1.89 美元/小时，训练 2–3 小时加上配置时间）。
**总时间：** 配置约 20 分钟，训练约 3 小时。

---

## 阶段 1——注册并充值

1. 在 https://runpod.io 注册（邮箱、密码、验证邮箱）。
2. 进入 Account → Billing，充值 10 美元。本项目有足够余量。

---

## 阶段 2——部署 Pod

1. 进入 **Pods → Deploy**。
2. **GPU：** 选择 **A100-SXM4-80GB**（约 1.89 美元/小时）或 **A100 PCIe 40GB**（约 1.29 美元/小时）。推荐 80GB，余量更大、训练更快。
3. **Container Image：** 在模板库搜索 **PyTorch 2.8**，或者直接填写：
   ```
   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
   ```
4. **Storage：** 容器磁盘 20 GB、volume 20 GB（默认值即可）。
5. **Expose HTTP Ports：** `8888`（仅在需要 Jupyter 时使用）。
6. 选择 **Deploy On-Demand**，不要选 Spot；Spot Pod 可能在训练中途被回收。

Pod 通常约 1 分钟启动。

---

## 阶段 3——连接

进入 **Pods → 你的 Pod → Connect → Start Web Terminal**。

现在你应当已经进入一个预装 PyTorch 2.8.0 的 Ubuntu 24.04 虚拟机，并拥有 root 权限。

---

## 阶段 4——检查初始环境

```bash
nvidia-smi | head -15
python3 --version
python3 -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.version.cuda}')"
```

预期结果：

- `NVIDIA A100-SXM4-80GB` 或类似型号
- `Python 3.11.x`
- `torch: 2.8.0+cu128, cuda: 12.8`

如果 torch 不是 2.8.0，说明选错模板：终止 Pod，并使用 `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu` 重新部署。

---

## 阶段 5——安装依赖（关键步骤）

```bash
pip install uv
uv pip install --system 'openpipe-art[backend]==0.5.17' 'pyyaml>=6.0' 'huggingface_hub'
```

原因：

- **`uv`：** Astral 的 pip 替代品，解析速度快，能正确处理定制 vLLM wheel URL。
- **`--system`：** 安装到系统 Python；单任务 Pod 不必额外维护虚拟环境。
- **`openpipe-art[backend]==0.5.17`：** `[backend]` extra 会拉取 ART 的定制 vLLM 分支（`vllm-0.17.0+art1`），以及固定版本的 trl、transformers、unsloth、accelerate、bitsandbytes 等。
- **`pyyaml` 与 `huggingface_hub`：** 本项目脚本直接使用，但不在 ART backend extra 中。

预计需要 5–10 分钟，并下载约 3 GB（包括定制 vLLM wheel）。已有 Ubuntu 包产生的黄色警告通常可以忽略；不要出现红色 traceback。

---

## 阶段 6——验证导入

```bash
python3 -c "import torch, art, vllm, trl, unsloth; print(f'torch={torch.__version__}, vllm={vllm.__version__}, trl={trl.__version__}, unsloth={unsloth.__version__}, all imports OK')"
```

关键的预期输出：

```text
torch=2.8.0+cu128, vllm=0.17.0+art1, trl=0.20.0, unsloth=2026.3.3, all imports OK
```

尤其检查 `vllm=0.17.0+art1`。如果是别的版本，说明安装成了上游 vLLM，而不是定制分支；可使用文末 Debug 方案强制安装。

---

## 阶段 7——克隆仓库

需要新的 GitHub PAT。建议专门生成一个新 token，不要复用曾经粘贴到聊天中的 token：

1. 打开 https://github.com/settings/personal-access-tokens/new。
2. 选择 **Fine-grained token**，有效期 1 天。
3. Repository access 只选择 `abhid1234/RL-Trained-Tool-Use-Agent`。
4. Repository permissions 只授予 Contents → Read-only。
5. 点击 Generate token，并复制只显示一次的 `github_pat_...` 值。

在 Pod 终端中：

```bash
cd /workspace
read -s GH_TOKEN
# 此时终端会等待输入；粘贴 token 后回车，输入不会显示。
git clone https://${GH_TOKEN}@github.com/abhid1234/RL-Trained-Tool-Use-Agent.git
cd RL-Trained-Tool-Use-Agent
unset GH_TOKEN
git log --oneline -3
```

---

## 阶段 8——运行测试

```bash
python3 -m pytest tests/ -q
```

预期：`32 passed`。

如果因 trl 0.20、transformers 5.2 或 ART 0.5.17 的 API 差异失败，查看下面 Debug 部分。

---

## 阶段 9——重新生成场景

```bash
python3 scripts/generate_scenarios.py
```

预期：

```text
Generated 200 training + 50 validation scenarios
```

---

## 阶段 10——Smoke test

```bash
rm -rf .art
python3 -m src.train --steps 2 --rollouts 32
```

按以下顺序检查输出：

1. `GRPO config: 4 scenarios/step × 8 rollouts/scenario = 32 rollouts/step`
2. 某处出现 `vllm 0.17.0+art1` 的引擎初始化信息。
3. `Starting from step 0`。
4. 四行组内方差日志，每行类似 `group N: mean=... std=X.XXX min=... max=...`。
5. `groups with variance: N/4`，其中 N 至少为 1，最好为 4。
6. **关键行：** 出现 `loss=X.XXX` 或其他真实训练日志，而不是 `SKIP: all rewards equal`。
7. `Training complete! Final step: 2`。

如果看到 `SKIP: all rewards equal`，说明 logprobs 传递仍有问题，查看 Debug 的问题 5。若出现真实 loss，才继续完整训练。

---

## 阶段 11——完整训练

确认阶段 10 成功后执行：

```bash
rm -rf .art
python3 -m src.train --config configs/train_config.yaml 2>&1 | tee training.log
```

A100 上预计约 3 小时。观察 `Avg reward` 是否大致从前 5 步的 -1.5，上升到第 20 步约 +1.5，最终第 50 步约 +3。

不要误以为关闭 Web Terminal 会停止训练；训练仍会继续，但你会看不到实时输出。需要离开时可打开第二个终端使用 `tmux attach`。

如果 `Avg reward` 连续 10 步以上保持 -3，应回看阶段 10 的输出，最可能是 `src/rollout.py` 需要适配当前 ART API。

---

## 阶段 12——评估并推送模型

```bash
# 在验证集上评估
python3 -m src.evaluate --scenarios data/scenarios_val.jsonl

# 推送到 HuggingFace
read -s HF_TOKEN
# 粘贴 HF WRITE token 后回车
export HF_TOKEN
python3 scripts/export_model.py \
    --checkpoint-dir ./checkpoints/qwen-0.5b-tool-agent \
    --repo-id abhidaas/qwen-0.5b-tool-agent-grpo \
    --merge-adapter
unset HF_TOKEN
```

HF token 需要 **Write** 权限：https://huggingface.co/settings/tokens → New → Write。

---

## 阶段 13——停止 Pod（控制成本）

RunPod 会按 Pod 运行的每一分钟计费：

1. **Pods → 你的 Pod → Terminate**，不要只点 Stop；Terminate 会同时释放存储。
2. 确认终止对话框。
3. 在 Account → Billing 检查费用是否符合预期（约 4–6 美元）。

---

## 成本估算

| 阶段 | 时间 | A100 80GB（1.89 美元/小时） |
|---|---:|---:|
| 1–3 注册、部署、连接 | 5 分钟 | 约 0.16 美元 |
| 4–6 环境检查、安装、验证 | 10 分钟 | 约 0.32 美元 |
| 7–9 克隆、测试、生成场景 | 3 分钟 | 约 0.09 美元 |
| 10 smoke test | 5 分钟 | 约 0.16 美元 |
| 11 完整训练 | 180 分钟 | **约 5.67 美元** |
| 12 评估和导出 | 5 分钟 | 约 0.16 美元 |
| **合计** | **约 3.5 小时** | **约 6.56 美元** |

充值 10 美元后约有 3.50 美元余量。使用 1.29 美元/小时的 A100 PCIe 40GB 时，总价约 4.48 美元。

---

## Debug 手册

### 问题 1：阶段 6 显示的不是 `vllm=0.17.0+art1`

uv 回退到了上游 vLLM，可强制安装定制 wheel：

```bash
uv pip install --system --reinstall \
  'https://github.com/vivekkalyan/vllm/releases/download/v0.17.0-art1/vllm-0.17.0%2Bart1-cp38-abi3-manylinux_2_31_x86_64.whl'
python3 -c "import vllm; print(vllm.__version__)"
```

### 问题 2：运行测试或 smoke test 时 `ImportError`

ART 0.5.17 可能移动或重命名了某些 API。先阅读完整错误，再检查：

- `tests/test_rollout.py` 在模块级 stub 了 `art`，需要按新 API 更新 stub 类。
- `src/train.py` 调用 `backend.train(model, train_groups, learning_rate=...)`；如果参数改名，需要同步调用。
- `src/rollout.py` 使用 `art.Trajectory(messages_and_choices=..., reward=..., metrics=...)`；若关键字参数改变，需要同步适配。

### 问题 3：smoke test 运行但打印 `SKIP: all rewards equal`

ART 0.5.17 可能期待另一种 Choice 对象结构。检查：

```bash
grep -rn "allow_training_without_logprobs\|_get_packed_tensors" /usr/local/lib/python3.11/dist-packages/art/
```

阅读 `backend.py` 的相关部分，确认 `_get_packed_tensors` 寻找的字段，再调整 `src/rollout.py`。

### 问题 4：完整训练 CUDA out of memory

Qwen 0.5B 加定制 vLLM 在 A100 80GB 上通常有很大余量。若仍 OOM：

- 将 `configs/train_config.yaml` 中 `num_generations` 从 8 降至 4。
- 将 `rollouts_per_step` 从 96 降至 48。
- 将 `max_completion_tokens` 从 512 降至 256。

### 问题 5：奖励长时间保持 -3

说明模型没有生成工具调用。检查：

- `src/rollout.py` 的 `SYSTEM_PROMPT` 是否足够清楚。
- rollout 中的 `temperature` 和 `top_p`，当前为 1.2 / 0.95，可尝试提高 temperature。
- 场景是否确实能用现有工具解决。

### 问题 6：训练在 0 附近徘徊

可能是学习率过低或 GRPO 更新太小，可尝试：

```bash
python3 -m src.train --config configs/train_config.yaml --lr 1e-4
```

---

## 为什么这个方案能工作（根因）

真正的关键在于 `openpipe-art` 并不是按通常方式声明依赖：

1. 不带 extra 的 `openpipe-art` 只安装客户端依赖，不安装 torch、vLLM、unsloth。这是有意设计，因为客户端可以连接远程 ART server。
2. `openpipe-art[backend]` 才会安装训练栈，而且会从 GitHub 直接 URL 安装定制 vLLM 分支。该分支包含 ART 专用补丁，上游 vLLM 没有这些 API。
3. ART 的 `art/vllm/patches.py` 会导入 `vllm.entrypoints.openai.chat_completion.protocol`，这个路径只有定制分支存在。上游 vLLM 0.7–0.16 都会在导入时失败。
4. `[backend]` 还固定了 trl 0.20.0、transformers 5.2.0、unsloth 2026.3.3 等版本，这正是 ART 维护者测试的组合。

---

## 来源

- [ART pyproject.toml（main 分支）](https://raw.githubusercontent.com/OpenPipe/ART/main/pyproject.toml)
- [OpenPipe/ART GitHub](https://github.com/OpenPipe/ART)
- [openpipe-art on PyPI](https://pypi.org/project/openpipe-art/)
- [ART 安装文档](https://art.openpipe.ai/getting-started/installation-setup)
- [RunPod PyTorch 2.8 模板](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8)
- [定制 vLLM wheel URL](https://github.com/vivekkalyan/vllm/releases/download/v0.17.0-art1/vllm-0.17.0%2Bart1-cp38-abi3-manylinux_2_31_x86_64.whl)

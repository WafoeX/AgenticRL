# RUNBOOK——运行与故障排查指南

本文记录在 Colab 和 RunPod 上运行项目时遇到的坑及修复方法。遇到异常时，先搜索这里。

---

## 🚨 已发现的根因（2026-04-12）

在 Colab 和 RunPod 上经历约 7 小时的依赖冲突后，真正的根因是：

> **ART 使用的是从 GitHub 直接 wheel URL 分发的定制 vLLM 分支，而不是 PyPI 上游 vLLM。**

定制分支是 `vllm-0.17.0+art1`，地址为 https://github.com/vivekkalyan/vllm/releases。它包含 ART 补丁所导入的 `vllm.entrypoints.openai.chat_completion.protocol` 模块。**任何 PyPI 上的上游 vLLM 都没有这个模块。**因此我们之前尝试的所有上游版本固定都会失败：ART 的代码无法使用它们。

下面所有问题，本质上都是同一个根因：我们固定了上游包，却没有使用声明定制 vLLM wheel 直接依赖的 ART `[backend]` extra。

## ✅ 正确安装方式

```bash
pip install uv
uv pip install --system 'openpipe-art[backend]==0.5.17' 'pyyaml>=6.0' 'huggingface_hub'

# 验证——vllm 必须是 0.17.0+art1
python3 -c "import torch, art, vllm, trl, unsloth; print(f'torch={torch.__version__}, vllm={vllm.__version__}, trl={trl.__version__}, unsloth={unsloth.__version__}, all imports OK')"
```

预期输出：

```text
torch=2.8.0+cu128, vllm=0.17.0+art1, trl=0.20.0, unsloth=2026.3.3, all imports OK
```

### 主机要求

- **Python 3.11+**（ART 的 pyproject 要求 `>=3.11`）。
- **预装 torch >= 2.8.0**。Colab 通常版本较旧，原地升级不稳定；RunPod 的 PyTorch 2.8 模板是已知可用起点。
- **Linux x86_64**，glibc ≥ 2.31（用于 `manylinux_2_31_x86_64` vLLM wheel）。

### `[backend]` 安装的版本

| 包 | 版本约束 |
|---|---|
| `torch` | `>=2.8.0` |
| `transformers` | `==5.2.0` |
| `trl` | `==0.20.0` |
| `peft` | `>=0.14.0` |
| `unsloth` | `==2026.3.3` |
| `unsloth-zoo` | `==2026.3.1` |
| `vllm` | GitHub 定制 wheel URL |
| `bitsandbytes` | `>=0.45.2` |
| `accelerate` | `==1.7.0` |

这些版本直接来自 [ART 的 pyproject.toml](https://raw.githubusercontent.com/OpenPipe/ART/main/pyproject.toml) 的 `[project.optional-dependencies].backend`。

---

## 已废弃的安装命令

早期命令如下，但现在**不能使用**：

```python
!pip install --quiet 'openpipe-art' 'trl>=0.15,<0.17' 'vllm>=0.10,<0.12' unsloth datasets peft pyyaml huggingface_hub
```

它会安装缺少 ART 所需模块的上游 vLLM。请改用上面的 `[backend]` extra。

---

## 已知问题与修复

### 问题 1：`ModuleNotFoundError: No module named 'unsloth'`

**原因：** openpipe-art 在模块加载时执行 `import unsloth`，但没有把 unsloth 声明为强制 pip 依赖；全新环境会缺失它。

**修复：** 显式安装 unsloth；正确安装命令已经包含它：

```python
!pip install unsloth
```

### 问题 2：`ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`

**原因：** `trl>=0.17` 移动或删除了 `trl.trainer.utils` 中的 `ConstantLengthDataset`，而 `unsloth_zoo/dataset_utils.py` 仍从旧位置导入。

**修复：** 原来的兼容办法是固定 `trl<0.17`；当前应优先使用 ART `[backend]` 的固定版本，不要自行混装：

```python
!pip install --force-reinstall --no-deps 'trl==0.15.2'
```

### 问题 3：`ModuleNotFoundError: No module named 'vllm.adapter_commons'`

**原因：** `vllm>=0.12`（尤其是 0.16+）删除了 `adapter_commons`，而 `unsloth_zoo/vllm_lora_request.py` 仍导入它。

**修复：** 历史方案是固定 `vllm<0.12`；当前应使用 ART 提供的 `0.17.0+art1` 定制分支，不能拿上游 0.10.2 替代：

```python
!pip install --force-reinstall --no-deps 'vllm==0.10.2'
```

### 问题 4：固定版本时 `ResolutionImpossible`

**原因：** pip 找不到同时满足所有相互约束的版本组合。

**修复：** 用 uv 和 ART `[backend]`，不要手工拼接训练栈。如果必须调查旧环境，可用范围约束而非精确版本，并避免同时强制 transformers：

```python
!pip install uv
!uv pip install --system 'trl<0.17' 'vllm<0.12' 'openpipe-art' unsloth datasets peft pyyaml huggingface_hub
```

### 问题 5：打印 `SKIP: all rewards equal`，但组统计有方差

**原因：** 真正触发条件是 `backend._get_packed_tensors()` 返回 `None`，最常见原因是轨迹中没有逐 token logprobs，而不一定是奖励真的完全相同。

**修复：** 确认 `src/rollout.py`：

1. 每次 `chat.completions.create` 都传入 `logprobs=True, top_logprobs=1`。
2. `messages_and_choices` 保存原始 OpenAI `Choice` 对象，而不是转换后的字典。

检查命令：

```python
!grep -c "logprobs=True" src/rollout.py
!grep -c "_build_api_messages" src/rollout.py
```

### 问题 6：模型输出确定，导致 “All rewards equal”

**原因：** temperature=1.0 时，Qwen 0.5B 基础模型对同一 prompt 几乎输出相同内容，组内方差为零，GRPO 没有学习信号。

**修复：** 使用 `temperature=1.2, top_p=0.95` 增加探索；当前 `src/rollout.py` 已采用此设置。

### 问题 7：vLLM 报 `Already borrowed`

**原因：** vLLM 0.16+ 的 guided tool-calling 存在并发 bug；并行请求会竞争 sampler 状态，LoRA 交换后更明显。

**修复：** 定制分支和当前 rollout 中都有兼容方案：对 `BadRequestError` 做有界重试，并在最终失败时返回 reward=-3.0 的失败轨迹，避免单条轨迹拖垮整个训练步骤。

### 问题 8：训练立即退出并显示 `Starting from step N / Training complete`

**原因：** ART 将检查点保存在 `.art/`；之前的运行（包括失败运行）提高了步数计数器。如果 `--steps N` 不大于当前计数，循环不会执行。

**修复：** smoke test 前清理 `.art`：

```python
!rm -rf .art
!python3 -m src.train --steps 2 --rollouts 32
```

### 问题 9：只有 1 个 group，训练被跳过

**原因：** 每步只有 1 组时，ART 无法计算跨组优势；至少需要 2 组，建议 4 组以上。

**修复：** 使用 `--rollouts 32` 或更大值。`num_generations=8` 时能得到至少 4 组：

```python
!python3 -m src.train --steps 2 --rollouts 32
```

### 问题 10：`Trajectory.metrics` ValidationError

**原因：** ART 的 pydantic `Trajectory.metrics` 严格要求 `dict[str, float | int | bool]`；旧代码把 `tools_called`（字符串列表）和 `task`（字符串）塞进 metrics。

**修复：** metrics 里只放数值。当前实现已经修复。

### 问题 11：`fatal: not a git repository`

**原因：** runtime 重启或超时断开清空了 `/content/`，仓库、`.art` 和本地修改全部消失；安装包是否保留也不确定。

**修复：** 重新克隆：

```python
from google.colab import userdata
gh_token = userdata.get('GITHUB_TOKEN')
!rm -rf /content/RL-Trained-Tool-Use-Agent
!git clone https://{gh_token}@github.com/abhid1234/RL-Trained-Tool-Use-Agent.git /content/RL-Trained-Tool-Use-Agent
%cd /content/RL-Trained-Tool-Use-Agent
!git log --oneline -3
```

### 问题 12：`SecretNotFoundError: Secret GITHUB_TOKEN does not exist`

**原因：** Colab 🔑 侧边栏中没有添加 secret、Notebook access 没打开，或大小写写错；正确名称是 `GITHUB_TOKEN`。

**修复：** 打开 🔑 侧边栏 → Add new secret → 名称 `GITHUB_TOKEN` → 填入 PAT → 打开 **Notebook access**。

### 问题 13：pip 的 `ResolutionTooDeep: 200000`

**原因：** pip 依赖解析器回溯 200,000 次后放弃，通常是依赖图过于纠缠。

**修复：** 使用 uv：

```python
!pip install --quiet uv
!uv pip install --system 'openpipe-art' 'trl<0.17' 'vllm<0.12' unsloth datasets peft pyyaml huggingface_hub
```

---

## 计算预算管理

### 初始预算

Colab Pro 每月约 100 compute units，价格 9.99 美元。

### A100 消耗速率

| 状态 | 速率 |
|---|---|
| A100 正在训练 | 约 11.77 units/小时 |
| A100 空闲但连接中 | 约 3–5 units/小时 |
| A100 已断开 | 0 units/小时 |

### 本项目预算

| 活动 | units |
|---|---:|
| Smoke test（Cell 6，约 3 分钟） | 约 1 |
| 完整训练（Cell 7，约 3 小时） | 约 36 |
| 评估（Cell 8，约 5 分钟） | 约 1 |
| HF 导出（Cell 9，约 2 分钟） | <1 |
| **总计** | **约 40** |

还剩约 60 units 用于重试和调试；预算不是本项目的主要瓶颈。

### 硬性保护措施

1. 每次会话开始检查右上角 RAM/Disk 下拉菜单中的 compute units；低于 40 时不要启动完整训练。
2. 离开时断开 runtime（Runtime → Disconnect），避免空闲消耗。
3. 定期进入 Colab 账户菜单中的 Manage Sessions，终止遗忘标签页产生的 ghost session。
4. 如果下个月不再需要 Colab Pro，在 28 天内取消自动续费。

---

## Runtime 状态管理

| 操作 | Python 状态 | `/content/` | 已安装包 | 适用场景 |
|---|---|---|---|---|
| **Restart session** | 清空 | 保留 | 保留 | Python kernel 行为异常但环境正常 |
| **Disconnect runtime** | 清空 | 保留 | 暂时保留 | 离开时节省空闲费用 |
| **Disconnect and delete runtime** | 清空 | 清空 | 清空 | 环境损坏，需要全新 VM |

不确定时，执行：**Disconnect and delete runtime → Reconnect → Cell 2 安装 → Cell 3 克隆**。

---

## 训练健康检查

### Cell 6 smoke test 之后

启动 Cell 7 之前确认：

1. GRPO 配置显示 `4 scenarios/step × 8 rollouts/scenario = 32 rollouts/step`。
2. 出现 4 行组内方差，每行 `std > 0`。
3. ART 没有只打印 SKIP；应看到 `loss=X.XXX`、`train/reward` 或类似内容。
4. 启动阶段没有 import error。
5. 少量 `[rollout] caught BadRequestError` 可以接受；如果超过约 10%，应重新检查 vLLM 定制分支和重试逻辑。

### Cell 7 运行期间

- `Avg reward` 应逐步上升。
- `groups with variance: N/12` 应大多数时候接近 12/12。
- `Avg tools/rollout` 应稳定在约 1.5–2.5，表示模型调用了合理数量的工具。
- Qwen 0.5B 在 A100 上通常不会 OOM；若发生，先把 `num_generations` 从 8 降至 4。

---

## 代码架构提醒

### `src/rollout.py`——agent 循环

每个 rollout 请求都必须包含：

- `logprobs=True`——ART 用它计算 GRPO 策略梯度。
- `top_logprobs=1`——至少保存被选 token 的 logprob。

每条 trajectory 都必须：

- assistant turn 保存原始 `Choice` 对象，不要转成字典。
- system/user/tool turn 使用普通字典。
- metrics 只放数值，不放字符串和列表。

### `src/train.py`——GRPO 循环

GRPO 要求：

- `rollouts_per_step % num_generations == 0`。
- 有多个 group，建议 `rollouts_per_step / num_generations >= 4`。
- 使用全新的 `.art` 状态，或保证 `num_steps > current_step`。

### `src/rewards.py`——分解式奖励

范围为 `[-3, +4]`：

- `r_format`（0 或 1）：是否产生工具调用。
- `r_correct`（-3 到 +3）：工具名、参数键和值的 Jaccard/精确匹配。

如果某个 group 的所有 rollout 得到同一奖励，GRPO 会跳过该组；当前 remedy 是提高 rollout 中的 temperature（1.2 + top_p 0.95）。

---

## 如果一切都失败

1. **进一步固定版本。** 可尝试 `trl==0.15.2`、`vllm==0.10.2`、`unsloth==2025.3.19`、`openpipe-art==0.3.2`，但这是历史排障方案，优先遵循当前 ART `[backend]`。
2. **切换到 RunPod。** 约 3–4 美元可租用 3 小时 A100，干净的 Ubuntu 环境能避开不少 Colab 特有问题。
3. **先休息，再继续。** 有时最好的调试动作是第二天再看。

---

## 变更日志

- **2026-04-12：** 初版运行手册；在经历一整天 Colab 调试、发现上述问题后编写。

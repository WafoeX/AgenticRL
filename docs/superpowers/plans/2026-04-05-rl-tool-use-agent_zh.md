# RL-Trained Tool-Use Agent——实现计划（中文翻译）

> **给智能体开发者：** 建议使用逐任务的 agentic 开发或执行计划流程。本文的步骤使用复选框（`- [ ]`）跟踪。原文中的代码、命令、类名和 API 名称保持不翻译，便于直接复制执行。

## 目标

使用 OpenPipe/ART 的 GRPO 训练 Qwen 2.5 0.5B，使它学会多步 API 调用，并在结构化工具使用任务上超过只依赖提示词的 7B 模型。

## 架构

在单张 GPU 上使用 ART client-server 的 LocalBackend colocate 模式。智能体在确定性的工具环境中执行 rollout；环境提供 5 个工具：计算器、天气、搜索、单位转换和数据库查询。每条 rollout 生成一个 `art.Trajectory`，使用分解式奖励评估格式、工具名和参数是否正确。每个 prompt 生成 8 条 rollout，经过 50 个步骤进行 GRPO 训练。

## 技术栈

Python 3.11、OpenPipe ART、HuggingFace TRL（由 ART 使用）、vLLM、Qwen 2.5 0.5B Instruct、PyTorch 2.4+、Google Colab A100。

---

## 文件结构

```text
rl-tool-agent/
├── README.md                      # 项目说明、安装和结果
├── requirements.txt               # 依赖
├── src/
│   ├── __init__.py
│   ├── tools.py                   # 5 个工具定义和确定性执行器
│   ├── scenarios.py               # 多步工具使用场景生成器
│   ├── rewards.py                 # 分解式奖励函数
│   ├── rollout.py                 # ART 多步 agent loop
│   ├── train.py                   # ART + GRPO 主训练脚本
│   └── evaluate.py                # 评估 RL 模型与提示基线
├── tests/
│   ├── test_tools.py              # 工具执行器测试
│   ├── test_rewards.py            # 奖励函数测试
│   ├── test_scenarios.py          # 场景生成测试
│   └── test_rollout.py            # mock rollout 集成测试
├── notebooks/
│   └── train_colab.ipynb          # Colab A100 训练 notebook
├── data/
│   ├── scenarios_train.jsonl      # 200 条训练场景
│   └── scenarios_val.jsonl        # 50 条验证场景
├── configs/
│   └── train_config.yaml          # 超参数
└── scripts/
    ├── generate_scenarios.py      # 生成场景 JSONL
    ├── demo_comparison.py         # 视频用并排演示
    └── export_model.py            # 导出到 HuggingFace Hub
```

---

## 任务 1：项目骨架与依赖

**涉及文件：** 创建 `requirements.txt`、`src/__init__.py`、`tests/__init__.py` 和 `configs/train_config.yaml`。

首先初始化 Git 仓库并声明依赖。训练配置包括：基础模型 `Qwen/Qwen2.5-0.5B-Instruct`、项目名、50 个训练步骤、每步 96 条 rollout、学习率 `5.0e-5`、每个 prompt 生成 8 条、最多 512 个 completion token，以及 50 条验证场景。

`src/__init__.py` 放置包说明字符串；`tests/__init__.py` 可以为空。完成后提交：

```bash
git add requirements.txt configs/train_config.yaml src/__init__.py tests/__init__.py
git commit -m "feat: project scaffold with deps and config"
```

---

## 任务 2：工具环境

**涉及文件：** `src/tools.py`、`tests/test_tools.py`。

先为 5 个工具编写失败测试，再实现工具：

- `calculator`：执行受字符白名单限制的数学表达式。
- `get_weather`：从固定天气数据集中读取城市、温度、天气状况和湿度。
- `search`：从固定知识库按关键词重叠返回相关片段。
- `convert_units`：支持摄氏/华氏、公里/英里、千克/磅、米/英尺等转换。
- `lookup_db`：从国家数据表读取人口、首都、面积和 GDP。

所有工具都使用 OpenAI function calling schema，并通过统一的 `execute_tool` 入口路由。工具不能依赖外部 API，这样训练奖励可复现。

先运行：

```bash
python -m pytest tests/test_tools.py -v
```

预期的初始失败是 `ModuleNotFoundError: No module named 'src.tools'`；实现后所有工具测试通过，再提交变更。

---

## 任务 3：奖励函数

**涉及文件：** `src/rewards.py`、`tests/test_rewards.py`。

奖励参考 ToolRL：

```text
R_total = R_format (0 或 1) + R_correct（[-3, 3]）
总范围：[-3, 4]
```

实现内容：

1. `jaccard`：计算两个集合的 Jaccard 相似度。
2. `parse_tool_calls_from_messages`：从 assistant 消息中提取工具名和 JSON 参数。
3. `compute_reward`：分别比较工具名、参数键和值，再归一化到奖励范围。

测试必须覆盖：完美匹配、工具正确但参数错误、错误工具、没有预测、多步完全正确和多步部分正确。使用：

```bash
python -m pytest tests/test_rewards.py -v
```

---

## 任务 4：场景生成器

**涉及文件：** `src/scenarios.py`、`tests/test_scenarios.py`、`scripts/generate_scenarios.py`。

场景模板包含自然语言任务、参数生成函数、期望工具列表和期望调用构造函数。模板应覆盖：

- 搜索 + 计算器的年龄计算；
- 天气 + 单位转换；
- 数据库查询 + 数学计算；
- 简单数学、天气、搜索、单位转换或数据库查询；
- 搜索 + 转换；
- 查询 + 计算 + 转换的三步链；
- 两个城市或两个国家的比较。

`generate_scenario(seed)` 必须确定性地生成单条场景；`generate_dataset(count, seed)` 通过连续 seed 生成批量数据。测试需要检查模板字段、返回结构、相同 seed 的一致性、不同 seed 的差异和数据集数量。

运行：

```bash
python -m pytest tests/test_scenarios.py -v
python scripts/generate_scenarios.py
```

脚本应输出 `Generated 200 training + 50 validation scenarios`，并写入 `data/scenarios_train.jsonl` 和 `data/scenarios_val.jsonl`。

---

## 任务 5：Rollout 函数（ART 多步 agent loop）

**涉及文件：** `src/rollout.py`、`tests/test_rollout.py`。

rollout 的一次循环如下：

1. 用 system prompt、用户任务和工具 schema 请求模型。
2. 保存 assistant 的原始 `Choice` 对象。
3. 解析并执行每个工具调用，把 tool 结果追加回对话。
4. 如果模型不再调用工具，或达到最大轮数，就结束。
5. 将实际工具调用与场景期望调用交给奖励函数。
6. 构造带有 reward 和数值 metrics 的 `art.Trajectory`。

这里有两个关键约束：

- 每次请求都必须传 `logprobs=True` 和 `top_logprobs=1`，否则 ART 无法构造 GRPO 的训练张量。
- `messages_and_choices` 中必须保留原始 `Choice`，不能为了下一次 API 调用而把它永久转成普通字典。

vLLM 的 `Already borrowed` 错误使用最多 3 次的指数退避重试；基础设施失败应返回 reward=-3.0 的失败轨迹，而不是让整个训练步骤崩溃。

运行：

```bash
python -m pytest tests/test_rollout.py -v
```

测试使用 mock 的 ART、OpenAI 客户端和模型，不需要 GPU；应验证单工具成功路径和 `MAX_TURNS` 上限。

---

## 任务 6：训练脚本（ART + GRPO）

**涉及文件：** `src/train.py`。

训练脚本应完成：

1. 从 JSONL 读取训练和验证场景。
2. 检查 `rollouts_per_step % num_generations == 0`。
3. 使用 `LocalBackend` 注册 `art.TrainableModel`。
4. 每步选择若干场景；同一场景生成 `num_generations` 条轨迹，构成一个 GRPO group。
5. 打印平均奖励、平均工具数和每组奖励方差。
6. 调用 `backend.train` 更新模型并写入训练日志。
7. 每 5 步使用验证场景评估。
8. 支持从检查点继续训练，以及命令行覆盖 steps、rollouts、learning rate 和 model。

ART 0.5.17 的 `_monitor_openai_server` 存在后台竞态：LoRA 注册尚未完成时监控请求可能返回 404，后台异常会进一步关闭 proxy。实现一个 `_safe_monitor` 包装器，只记录异常而不让其传播到训练主循环。

导入检查：

```bash
python -c "from src.train import train; print('train.py imports OK')"
```

本地未安装 ART 时导入可能失败，这是环境问题，不是训练逻辑测试失败。

---

## 任务 7：评估脚本

**涉及文件：** `src/evaluate.py`。

定义 `EvalResult` 和 `EvalReport` 两个 dataclass。单场景评估要用温度 0 运行模型，执行与训练相同的工具循环，但不更新梯度。报告提供：

- 成功率（reward > 2.0）；
- 平均奖励；
- 每个场景调用过的工具；
- RL 训练模型与提示基线的差值。

结果写入 `data/eval_results.json`。脚本通过 OpenAI 兼容的 `base_url` 连接已启动的模型服务。

---

## 任务 8：演示对比脚本

**涉及文件：** `scripts/demo_comparison.py`。

为一分钟视频准备固定任务，包括：

- 搜索 + 计算的两步链；
- 天气 + 转换的两步链；
- 查询 + 计算 + 转换的三步链。

分别调用 RL 训练模型和提示基线，输出彩色 PASS/FAIL、奖励、工具调用顺序和最终胜场统计。

---

## 任务 9：HuggingFace 导出脚本

**涉及文件：** `scripts/export_model.py`。

从检查点加载 tokenizer 和模型；当传入 `--merge-adapter` 时加载 Qwen 基础模型、载入 LoRA 适配器并调用 `merge_and_unload()`。随后通过 `push_to_hub` 推送模型和 tokenizer。

典型用法：

```bash
python scripts/export_model.py \
    --checkpoint-dir ./checkpoints/qwen-0.5b-tool-agent \
    --repo-id yourusername/qwen-0.5b-tool-agent-grpo \
    --merge-adapter
```

---

## 任务 10：Colab 训练 Notebook

**涉及文件：** `notebooks/train_colab.ipynb`。

Notebook 应按顺序提供以下单元：

1. Markdown：项目说明和 GPU 检查。
2. 安装 ART backend、PyYAML 和 HuggingFace 工具。
3. 克隆仓库并确认当前目录。
4. 验证 torch、vLLM、TRL、Unsloth 版本。
5. 应用 `scripts/patch_art_tokenize.py` 并生成场景。
6. 删除 `.art` 后执行 2 步、32 rollout 的 smoke test。
7. 运行完整训练、评估并导出模型。

由于 torch 2.8 的要求，Notebook 是备用方案；推荐优先使用 `RUNPOD.md`。

---

## 任务 11：README

**涉及文件：** `README.md`。

README 应解释实验结果、架构、快速开始、依赖陷阱、RunPod/Colab 运行方式、项目结构、技术栈、论文和产物。至少提供以下快速开始命令：

```bash
pip install -r requirements.txt
python scripts/generate_scenarios.py
python -m src.train --steps 2 --rollouts 8  # smoke test
```

源码说明应覆盖：

```text
src/tools.py       —— 5 个确定性工具定义和执行器
src/scenarios.py   —— 多步场景生成器
src/rewards.py     —— 分解式奖励函数
src/rollout.py     —— 模型与工具之间的 agent loop
src/train.py       —— ART LocalBackend 上的 GRPO 循环
src/evaluate.py    —— RL 与提示基线的并排评估
```

---

## 任务 12：运行全部测试和最终验证

运行：

```bash
python -m pytest tests/ -v
```

预期所有测试通过。随后检查项目文件：

```bash
find . -type f ! -path './.git/*' ! -path '*__pycache__*' | sort
```

最终验证内容包括：依赖可导入、场景 JSONL 数量正确、工具和奖励单元测试通过、mock rollout 通过、smoke test 产生非零训练信号、训练和评估脚本能够按文档启动。

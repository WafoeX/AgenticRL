# RL-Trained Tool-Use Agent

**基于 Qwen2.5-0.5B 的工具调用强化学习实验：训练复现、评估稳定化与轨迹级奖励工程。**

项目基于既有开源工程，使用 OpenPipe ART、GRPO 和 LoRA 构建工具调用训练链路。当前重点是让评估结果可信：区分基础设施失败与模型表现、明确任务级和轨迹级指标，并针对缺步调用和动态参数实现可选 Reward V2。

> **当前进度：Phase 1/2 已完成服务器验收；Phase 3 代码和测试用例已实现，尚待服务器验收。**
> 当前实验来自 **step 2 checkpoint**。没有已验证的 V2 训练收益，也没有完成 50-step 训练的证据。

## 项目概览

| 项目 | 内容 |
|---|---|
| 基础模型 | Qwen2.5-0.5B-Instruct，约 0.5B 参数 |
| 学习方法 | GRPO，按同一场景分组采样；ART 后端管理 LoRA 训练 |
| 工具环境 | 5 类本地确定性工具：计算器、天气、搜索、单位转换、国家数据查询 |
| 数据规模 | 12 类模板，200 条训练场景、50 条验证场景 |
| 任务深度 | 预期 1–3 次工具调用；Agent 最多交互 6 轮 |
| 核心改造 | 只读诊断、双模式评估、分批限并发、可选轨迹级奖励与完整链路审计 |

这些工具使用本地固定数据，不调用真实天气或搜索服务；项目是可控的 RL 实验环境，不是生产级外部 API Agent。

## 实验结果

### 1. 评估稳定性改善

两次评估均使用同一个 **step 2 checkpoint**，50 个场景、每场景采样 8 条轨迹，共 400 次尝试。

| 指标 | Phase 1 | Phase 2 sample | 变化 |
|---|---:|---:|---|
| 已标记基础设施失败 | 180/400（45%） | 0/400（0%） | **下降 45 个百分点** |
| 未标记失败的轨迹 | 220/400 | 400/400 | 增加 180 条，**相对增加 81.8%** |

Phase 2 将独立评估改为顺序场景批次，并分离生成参数、关闭评估 logprobs。默认每批 4 个场景，sample 模式的瞬时 rollout 上界为 `4 × k`，k=8 时不超过 32。

**这是评估工程的观测改善，不是模型学习提升。** 未标记失败不等于任务正确；两次运行同时改变了多项评估设置，不能把效果全部归因于某一个措施，也不意味着以后不会出现基础设施错误。

### 2. 当前 checkpoint 的评估表现

以下数字均采用 **V1 的 `reward > 2.0` 代理成功判定**，不是最终答案正确率，也不是完整工具链成功率。

| 指标 | deterministic | sample（k=8） |
|---|---:|---:|
| 场景数 / rollout 数 | 50 / 50 | 50 / 400 |
| 场景级奖励阈值成功率 | **22%（11/50）** | **观测 Pass@8：30%（15/50）** |
| 轨迹级奖励阈值成功率 | 22%（11/50） | **21%（84/400）** |
| 平均 reward | +0.137 | -0.115 |
| No-tool rate | 6% | 14% |
| 已标记基础设施失败率 | 0% | 0% |

- **Deterministic：**每场景一次尝试，temperature=0、top-p=1。
- **观测 Pass@8：**每场景 8 次采样中，至少一次达到成功阈值的场景比例。
- **Rollout 成功率：**成功轨迹数 / 全部轨迹数，不能当作 Pass@8。
- **基础设施失败：**按现有 `failed == 1` 标记统计；失败尝试保留在总分母，不静默删样本或重采样。

结果整理自此前用户服务器反馈，详见 [Phase 2 验收记录](docs/phase2_evaluation_zh.md)。原始服务器评估 JSON 尚未存入当前仓库；本 README 未重新运行实验。`temperature=0` 也不保证跨 GPU、版本和内核逐字节复现。

### 3. 为什么这些数字不能证明多步能力

step 2 的确定性评估中，50 条轨迹共计 47 次工具调用，3 条无工具调用，即每条非空轨迹都只调用一次工具。3 个需要两次 `lookup_db` 的 `country_compare` 场景，却被旧奖励阈值记为成功：**这是相对预期完整工具链的缺步假阳性。**

因此，当前不能把奖励阈值下的两步成功当作真正的完整链路能力，也不能将 README 旧版中的上游 step 15 结果与本项目 step 2 直接比较。V2 能否改善训练仍需固定模型、数据和评估标准的受控对照。

## 主要技术亮点

### 只读诊断：先分清“服务失败”与“模型失败”

`src/diagnostics.py` 旁路统计全部/有效轨迹平均奖励、no-tool rate、已标记基础设施失败率和组内奖励标准差。诊断不改变原训练组成员、奖励或 GRPO 输入；零方差组只做观测，不自动重采样。

### 评估重构：明确分母，控制并发

`scripts/eval_checkpoint.py` 提供 deterministic、sample、both 三种模式；both 注册一次模型后顺序执行两种评估。报告按任务类型和预期调用次数分层，重复工具名也计入深度。缺少场景组、尝试或发生轨迹对齐异常时立即报错，避免缩小分母产生虚高结果。

### Reward V2：从单调用相似度到轨迹约束

V2 为显式启用功能，默认仍使用 V1。奖励保持在 `[-3, +4]`，分项如下：

| 分项 | 权重/满分 | 作用 |
|---|---:|---|
| Baseline | -3 | 保持奖励量级 |
| Format | 1 | 校验已知工具名和参数字典 |
| Tool multiset | 1 | 保留重复调用的计数 |
| Sequence | 1.5 | 综合前缀、完整序列及调用数量 |
| Argument keys | 1 | 匹配参数键 |
| Static + dynamic values | 合计 1.5 | 参数归一化与 observation 依赖匹配 |
| Completion | 1 | 区分部分执行与完整链路 |

正确前缀获得部分 credit；未完成链路的 reward 额外限制为 **≤2**，避免再次超过旧成功阈值。数值使用 Decimal 归一化，数学表达式通过受限 AST 比较格式，不执行表达式、不做任意符号代数。

### 动态依赖与同轨迹审计

旁路 trace 保存 `{name, arguments, result, turn}`，检查参数是否匹配前一步有效结果。实际 rollout 的依赖消费调用必须处于更晚的 assistant turn，避免将同一回复里预先生成的调用误当成 observation 复用。

`--compare-rewards` 对同一 trace 计算 V1/V2，不重复生成或执行工具；trace 按 trajectory 对象身份绑定，避免异步或组内重排导致错配。独立 **`complete-tool-chain-v1`** 判定调用数量、顺序、参数及有效执行结果，但**不判断最终答案正确性**。

上述 V2 与审计功能已实现，尚未完成服务器测试及 GPU 推理验收，详见 [Phase 3 设计与验收说明](docs/phase3_reward_v2_zh.md)。

## 工作流程

```text
任务 + 工具 Schema
       ↓
Qwen / vLLM 生成工具调用
       ↓
确定性工具执行 → observation 回传 → 下一轮生成
       ↓
工具调用序列 + 可选执行 trace
       ├─ 训练：Reward → ART TrajectoryGroup → GRPO / LoRA 更新
       └─ 评估：分批采样 → 指标与诊断 → 可选完整链路/双评分 → JSON
```

训练保留原始 Choice 和 token logprobs，避免丢失 ART 构建训练数据所需的信息；独立评估使用同一个 rollout loop，但不请求 logprobs。

## 使用方式

**Colab A100 请优先使用 [Phase 3 独立验证指南](docs/colab_phase3_validation_zh.md)**：包含 SSH 拉取仓库、Drive 留档、隔离环境、原 checkpoint 恢复/新 smoke 分支、实验命令及 SSH 推送反馈。小型结果提交到独立 `feedback/<run-id>` 分支，大型 trace/checkpoint 留在 Drive，后续按清单下载分析。旧 notebook 含清空 `.art` 和 50-step 训练步骤，不适合直接用于本轮验收。

### 环境准备

建议在独立的 Linux/CUDA GPU 环境中准备 Python 3.11+、PyTorch 2.8+ 及 ART 后端：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 固定 `openpipe-art[backend]==0.5.17`。已有可运行服务器不应为阅读或评估而盲目升级依赖。历史部署和兼容性问题见 [RUNPOD_zh.md](RUNPOD_zh.md)、[RUNBOOK_zh.md](RUNBOOK_zh.md)。

**版本提示：**历史文档记录定制 `vllm-0.17.0+art1`，而查阅的 ART `v0.5.17` 官方依赖声明 `vllm==0.17.0`。具体 wheel 来源、补丁适用性及实际 GPU 环境应以服务器安装记录为准；不要仅凭旧说明强制替换已工作的环境。历史价格、耗时和测试数量也不是当前验收结果。

### 优先评估已有 checkpoint，不重新训练

在保存 `.art` 的项目目录及原训练身份配置下运行：

```bash
python scripts/eval_checkpoint.py \
  --config configs/train_config.yaml \
  --scenarios data/scenarios_val.jsonl \
  --mode both --generations 8 \
  --reward-version v1 \
  --output eval_both.json
```

脚本加载该模型身份的当前 checkpoint。**先核对日志中的 `Loaded checkpoint: step ...`**；没有 checkpoint 时不能把注册后的基础模型当作本文 step 2 结果的复现。

- 双模式结果位于 `evaluations.deterministic` 和 `evaluations.sample`。
- 顶层 `avg_reward` 指 sample 平均奖励。
- 如服务不稳定，可增加 `--batch-scenarios 1` 降低并发。
- 本入口不调用训练、不主动删除 checkpoint；保留现有 `.art`，不要为评估重新生成数据。

### Phase 3 服务器验收（待执行）

先运行当前完整测试集；旧阶段的 PASS 不代表新增代码已经通过：

```bash
python -m pytest tests/ -v
```

测试通过后，在同一已有 checkpoint 上执行 V1 主评分的双模式审计：

```bash
python scripts/eval_checkpoint.py \
  --config configs/train_config.yaml \
  --mode both --generations 8 --reward-version v1 \
  --complete-chain --compare-rewards --save-traces \
  --output eval_step2_reward_audit.json
```

再验证 V2 rollout 集成，不更新模型：

```bash
python scripts/eval_checkpoint.py \
  --config configs/train_config.yaml \
  --mode deterministic --reward-version v2 \
  --complete-chain --compare-rewards --save-traces \
  --output eval_step2_v2_deterministic.json
```

`--reward-version v2` 表示本次采用 V2 **评分**，不证明加载的 checkpoint 已经过 V2 **训练**。完整链路指标默认关闭；验收标准及字段解释见 [Phase 3 文档](docs/phase3_reward_v2_zh.md)。

## 当前进度与限制

| 阶段 | 工作 | 状态 |
|---|---|---|
| Phase 1 | 只读诊断与失败拆分 | 服务器 `49 passed`，已有 step 2 评估 |
| Phase 2 | 参数分离、指标拆分、分批限并发 | 服务器确认全套 PASS；确定性和采样评估完成，最终测试数量未记录 |
| Phase 3 | Reward V2、完整工具链、同 trace 双评分 | 代码和测试用例完成，服务器验收待办 |
| Phase 4 | 训练失败样本隔离、异常精确归因 | 未实施 |
| Phase 5 | Best checkpoint、early stopping | 未实施 |

当前开发与训练暂停，优先保留和整理已有证据。尚不能声称模型准确率提升、推理加速或训练成本下降。

已知边界：

- 验证集是模板化实验数据，**50 条中有 36 条的任务文本在训练集中出现过**；不同 seed 不代表严格独立的泛化评估。
- 部分模板存在知识缺失和平方单位转换等语义问题，预期工具协议不等于最终答案正确。
- 完整链路 checker 是严格协议检查，不接受所有语义等价解法或独立调用重排；例如直接查询 Fahrenheit 可以回答问题，但不符合指定的两步协议。
- 现有失败标记和兼容性 workaround 不代表全部后端异常已解决；训练中的失败样本隔离尚未实施。
- 当前 YAML 的 `num_steps: 50` 是配置目标，不是完成进度；实际训练生成参数以 `src/train.py` / `src/rollout.py` 传递值为准，并非所有 YAML 字段均接入运行逻辑。

## 项目结构与文档

```text
src/
  tools.py                 # 本地确定性工具与 OpenAI function calling schemas
  scenarios.py             # 12 类参数化场景模板
  rewards.py               # 原 V1 奖励及版本入口
  reward_v2.py             # 轨迹奖励、参数归一化、完整链路 checker
  rollout.py               # Agent loop、原始 Choice/logprobs、可选 trace
  diagnostics.py           # 只读诊断、类型/深度及完整链路统计
  train.py                 # ART / GRPO 训练入口
scripts/
  eval_checkpoint.py       # 当前独立 checkpoint 评估与审计入口
  generate_scenarios.py    # 场景生成（会写入数据文件）
  patch_art_tokenize.py     # 历史 ART 兼容性补丁
  export_model.py          # LoRA 合并与导出
configs/train_config.yaml  # 模型身份、训练和评估配置
data/                     # 200 条训练、50 条验证场景
tests/                    # 单元与集成回归用例
docs/                     # 阶段记录与项目介绍
```

- [Phase 2 评估与验收](docs/phase2_evaluation_zh.md)
- [Phase 3 Reward V2 与验收](docs/phase3_reward_v2_zh.md)
- [Colab A100 验证步骤与实验命令](docs/colab_phase3_validation_zh.md)
- [简历项目介绍、亮点与证据边界](docs/resume_project_intro_zh.md)
- [中文部署指南](RUNPOD_zh.md) / [英文部署指南](RUNPOD.md)
- [中文故障记录](RUNBOOK_zh.md) / [英文故障记录](RUNBOOK.md)
- 历史 README：[README_OLD.md](README_OLD.md)、[README_zh_OLD.md](README_zh_OLD.md)，原样保留；其图表和上游实验不代表当前 checkpoint。

## 技术栈与参考资料

**技术栈：** Python · asyncio · Qwen2.5-0.5B-Instruct · GRPO · LoRA · OpenPipe ART 0.5.17 · vLLM · PyTorch / TRL / Unsloth / Transformers · JSONL / YAML · pytest / pytest-asyncio。

- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)：分解式工具奖励的设计参考。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 原始方法与组内相对奖励。
- [ART v0.5.17](https://github.com/OpenPipe/ART/tree/v0.5.17)：训练编排、Trajectory 与 LoRA 管理。
- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)：基础模型信息。

本项目的贡献定位为**开源复现和工程改造**，不是从零实现上述框架。原实验作者的 [文章](https://abhid.substack.com/p/i-tried-to-train-a-05b-model-to-use) 与 [Hugging Face 模型/产物](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo)仅作为背景资料，不冒称为本项目当前 step 2 或 V2 的实验产物。

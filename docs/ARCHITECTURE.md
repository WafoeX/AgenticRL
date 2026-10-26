# 架构与训练方法

AgenticRL 在本地确定性工具环境中研究小模型的多步任务完成能力。最终研究路径是 **Qwen2.5-0.5B-Instruct → LoRA SFT → 三个独立 seed 的答案主导 GRPO → 冻结四模型评测**，使用 PyTorch、Transformers 与 PEFT，而非早期 ART/vLLM 后端。

## 系统结构

```text
冻结数据 / 配置 / 身份与源码锁
              ↓
任务 + 工具 schema → 模型生成 assistant 动作
                           ↓
                 确定性工具执行 / observation
                           ↓
                  下一轮动作或最终答案
                           ↓
              流程 checker + 独立答案 checker
                  ├─ train：奖励 → GRPO 更新
                  ├─ validation：固定规则选点
                  └─ test：冻结后同题评测与配对统计
```

天气、搜索、计算、单位转换与国家查询均为本地实现，不依赖真实外部服务。工具 observation 是环境反馈，不作为模型生成动作参与策略损失。

## 模块职责

| 层 | 主要文件 | 职责 |
|---|---|---|
| 数据 | `src/research_generator.py`、`research_schema.py`、`research_audit.py` | 场景生成、结构校验、容量与泄漏审计 |
| 环境 | `src/research_environment.py` | 固定事实、工具 schema 与确定性执行 |
| 判定 | `src/research_checker.py`、`research_answers.py` | 参考链、语义路线、动态依赖及最终数值/单位/实体检查 |
| SFT | `src/research_sft.py`、`research_sft_run.py` | train-only 专家轨迹、assistant-only 监督、LoRA 保存与验证 |
| 奖励 | `scripts/research_answer_grpo_v1.py` | 最终答案主奖励及有界流程辅助 |
| 策略更新 | `scripts/research_grpo_native.py` | 组内优势、动作 mask、概率比与 clipped 策略更新 |
| 训练编排 | `scripts/train_research_answer_grpo_v1.py` | 三 seed 独立初始化、固定采样日程、验证选点与保存 |
| 评测 | `scripts/eval_research_final_test_v1.py` | 四模型身份核对、固定任务、逐题保存、失败停止 |
| 统计 | `scripts/summarize_final_test.py` | 仅用逐题 CSV 复算已发布统计，不重新调用模型 |

`src/train.py`、`rollout.py` 等早期 ART 路径仍保留用于实现追溯，但不是最终实验的运行入口。两套依赖环境不可混装。

## 答案主导奖励

```text
A = 完整结束且最终答案数值、单位、协议正确 ? 1 : 0
F = clip((原流程分 + 3) / 7, 0, 1)
R = A + 0.2 × F
```

正确答案奖励在 `[1, 1.2]`，错误答案最多 `0.2`，避免“流程完整但最终答案错误”获得最高奖励。合法替代路线不因 strict 参考链不匹配而丢失答案主奖励。没有奖励分项消融，不能把整体收益归因于某一个分项。

## 训练与选点

- 数据固定为 train 1200 / validation 180 / test 240；专家监督和奖励只使用 train，validation 只用于开发与选点。
- 三个 RL seed：20261011、20261012、20261013，分别从同一 SFT30 和全新 AdamW 开始，不相互续训。
- 每 seed 固定 60 个计划批次，每批 2 个场景、每场景 4 条 rollout；三个 seed 使用相同的问题顺序。
- 采用多轮 assistant-only 动作 mask、clip 0.2、样本标准差加 `1e-4` 的优势归一化、学习率 `5e-6`、beta 0、每批单次更新。采样 logprob 作分母，teacher 快照用于漂移核验；两者差异是方法局限。
- 零信号批次不更新，也不补抽制造信号；三个 seed 最终有效更新为 28 / 31 / 31 次，不能将 60 个计划批次称为 60 次有效更新。
- 计划批次 20 / 40 / 60 各评完整 validation，按答案正确数、动态依赖有效数、较早批次依次选点，保留原 SFT 基线回退候选。
- test 前冻结四模型身份与评测协议；主比较为三个 RL seed 平均正确率减同一 SFT30 正确率，不事后挑最高 seed。

## 评测与完整性边界

答案正确率、strict、semantic 和动态依赖分别报告。strict 只检查参考链，不等于任务成功。主配对区间以 240 个同题差值重采样，条件于四个固定模型，不覆盖全部训练随机性。

源码/数据锁只能证明相对于锚点的字节一致性，不是签名或外部盲测认证。历史提交 ID 是原实验元数据；独立复现应以保留的文件 SHA-256 为身份依据，不要求旧提交仍能 checkout。

已知限制包括：训练日志仅保存 1436/1440 条（seed11 缺 4 条，未补造）、采样/teacher logprob 差异、仅一个 SFT seed 与三个 RL seed、共享部分事实/模板的合成基准。结果和成本见[最终报告](research_final_test_results_v1_zh.md)，依赖与可复现范围见[复现说明](REPRODUCIBILITY.md)。

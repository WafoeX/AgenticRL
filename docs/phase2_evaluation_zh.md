# Phase 2：评估参数分离与稳定化

## 状态

- Phase 1：用户服务器验收 `49 passed`，并已完成 step 2 的实际 checkpoint 评估。
- Phase 2 首次服务器测试：`70 passed, 1 failed`。失败项的共享 gather 替身逐场景执行，峰值并发为 8；测试错误地把 `4 × 8 = 32` 的并发上界要求成必须达到的峰值。
- 本次仅修测试与验收说明：显式使用串行/并行两种 gather 替身，验证批次不重叠、并发不超过上界、每批尝试完整，以及不同场景 reward 与输出顺序对齐。**修复后的测试尚未运行，仍待服务器复测**；没有改评估或训练运行逻辑。
- 不启动 Phase 3，不删除 `.art`，不运行大规模训练，不生成虚构实验结果。

## 1. 文件范围

- `src/rollout.py`：增加可选生成参数，默认训练行为不变。
- `src/diagnostics.py`：区分 rollout 成功与场景成功，按预期工具调用次数统计深度。
- `scripts/eval_checkpoint.py`：确定性/采样/双模式、限并发、报告与 JSON。
- `configs/train_config.yaml`：增加可选评估配置及默认值。
- `tests/test_evaluation.py`：新增 Phase 2 回归测试。
- `tests/test_phase1_integration.py`：更新评估参数断言；保留旧结果字段、训练输入和关闭诊断的兼容性检查。
- `src/train.py`：只整理导入、移除未使用导入和无占位符 f-string；训练逻辑未改。

## 2. 为什么修改

Phase 1 服务器评估中，400 次 rollout 有 180 次被标记为基础设施失败（45%）。原报告的 14.8% 是 `59/400` 的采样 rollout 成功率，**不是 Pass@8，也不是确定性 task accuracy**。有效 rollout 的 no-tool rate 为 11.4%，不能仅凭此次结果判定 collapse。

Phase 2 将采样策略与训练分开，并限制场景并发；不宣称已经消除基础设施错误。

| 用途 | Temperature | Top-p | Logprobs | 每场景尝试数 |
|---|---:|---:|---|---:|
| 默认训练 | 1.2 | 0.95 | True，top_logprobs=1 | 原训练配置 |
| deterministic | 0.0 | 1.0 | False，不传 top_logprobs | 1 |
| sample | 1.2 | 0.95 | False，不传 top_logprobs | 默认 8 |

`--mode both` 注册/加载一次模型，对同一 checkpoint 顺序执行 deterministic 和 sample。

默认每批 **4 个场景**，完成当前批才开始下一批。sample 模式的瞬时 rollout 上限是 `4 × generations`，不是 4 条 rollout；如果服务端仍不稳定，可降为 `--batch-scenarios 1`。

## 3. 主链路与兼容性

仍然使用同一个 `rollout → tool → compute_reward → art.Trajectory` 执行链，不复制 Agent loop，不改变发送给 ART 的原始 Choice 对象。

- `compute_reward` 完全未改；本阶段仍使用 v1。
- 训练采样、TrajectoryGroup、GRPO、LoRA 和 ART checkpoint 管理未改。
- 训练内每 5 step 的 validation 保持原行为；本阶段清理的是独立 checkpoint 评估入口。
- Scenario JSONL 不变；无 ID 的旧数据仍按 `unknown` 分类。
- 深度按 `len(expected_calls)` 计算，重复工具名也分别计数。
- 旧 `--generations 8` 命令默认仍进入 sample 模式。
- sample 报告的旧 `accuracy_pct` 保留所有 rollout 分母；新增 `pass_at_k_pct` 以场景为分母。
- `--no-diagnostics` 关闭诊断/任务类型/深度明细，但仍显示模式的核心指标和基础设施失败率。
- `--batch-scenarios` 可调整限并发；旧 YAML 缺少新增字段不会报错。

### 指标含义

所有成功判定目前仍是 **reward > 2.0**，不是最终答案 correctness；没有引入 LLM Judge。

- Deterministic Accuracy：一次确定性 rollout 成功的场景比例。
- Pass@k：k 次采样中至少一次成功的场景比例（实际观测值）。
- Sampled rollout success rate：成功 rollout / 所有 rollout，不能当作 Pass@k。
- Average Reward：所有尝试的平均 reward，保留旧分母；另报告 `valid_avg_reward`。
- Infra failure rate：`failed == 1` 的尝试比例；这些尝试不会被视为 task success。

基础设施失败仍保留为失败尝试，不悄悄重采样或缩小分母。发生失败时报告明确警告：task success 是受缺失尝试影响的保守观测值，all-attempt mean reward 也被污染。若 ART 返回缺失场景组或缺失尝试，评估直接失败，不输出缩小分母后的伪成功报告。

## 4. 测试覆盖

新增测试覆盖：

- 默认训练参数与逐次请求中的 logprobs 完全保留。
- 两种评估模式、每场景实际尝试次数、关闭 logprobs 时省略 top_logprobs。
- 原始 Choice 保存、同一 rollout loop 的工具执行与 reward。
- Pass@k 与 rollout success 分母的区别。
- 重复工具调用的深度计算。
- 4 场景顺序批处理：串行与并行 gather 分别覆盖，验证并发上限、批次不重叠、每批调用次数和场景结果顺序；不假设实际 ART 调度必须达到上界。
- both 模式只注册一次 checkpoint、不训练、不删除 checkpoint。
- no-tool 与 infrastructure failure 的区分。
- 不完整 gather、非法参数、空数据、旧 YAML 默认值与 CLI 覆盖。

新增用例尚未在本机运行；不以编辑器自动静态检查代替测试验收。

## 5. 服务器验收

先运行所有测试：

```bash
python -m pytest tests/ -v
```

然后在现有 step 2 checkpoint 上评估，不重新训练：

```bash
python scripts/eval_checkpoint.py \
  --scenarios data/scenarios_val.jsonl \
  --mode deterministic \
  --output eval_deterministic.json

python scripts/eval_checkpoint.py \
  --scenarios data/scenarios_val.jsonl \
  --mode sample --generations 8 \
  --output eval_sample.json
```

也可一次生成双模式报告：

```bash
python scripts/eval_checkpoint.py \
  --scenarios data/scenarios_val.jsonl \
  --mode both --generations 8 \
  --config configs/train_config.yaml \
  --output eval_both.json
```

双模式 JSON 的两份报告在 `evaluations.deterministic` 和 `evaluations.sample`；顶层 `avg_reward` 明确指 sample 的平均值。

验收重点：50 个场景应分别得到 50 条 deterministic rollout 和 400 条 sample rollout；按深度/任务类型的场景数之和应为 50；日志显示连续场景批次；检查 infra failure 是否下降，不能预先假定下降。

## 6. 仍未解决

- Dynamic 参数、调用顺序和重复工具 credit：Phase 3 的 opt-in reward v2。
- 基础设施失败进入训练 GRPO、空 choices 的异常归因：Phase 4。
- best validation step、可选 early stopping：Phase 5。
- 当前 reward 阈值不能证明最终答案正确；v1 的动态参数失配仍影响多步成功率。
- 固定 temperature=0 不能保证跨 GPU、vLLM 版本和数值内核逐字节一致；比较实验仍需固定环境、数据与 checkpoint。

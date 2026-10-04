# Phase 3：可选 Reward V2 与完整工具链评估

## 状态与边界

- Phase 2 已由用户服务器确认测试全通过，sample / deterministic 实测均无已标记的 infra failure。
- Phase 3 的代码与测试用例已补齐；**修改后的测试、GPU 推理尚未运行，待用户服务器验收**。不以编辑器自动静态检查当作测试通过。
- 不运行本机测试/训练，不删除 `.art`，不开始 Phase 4，不启动 10/50-step 实验。

## 1. 修改文件

- 新增 `src/reward_v2.py`：纯标准库评分、归一化、完整工具链 checker、未启用的 final-answer 接口。
- `src/rewards.py`：保留原 `compute_reward` 函数体，增加同一函数的 `compute_reward_v1` 别名、统一版本入口。
- `src/rollout.py`：增加 opt-in v2 和旁路 `trace_sink`；v1 的 ART metrics、采样参数和消息结构保持不变。
- `src/train.py`：只增加版本配置/CLI/验证及 rollout 参数传递，不改 GRPO、分组、场景顺序、更新和 checkpoint 管理。
- `src/diagnostics.py`：增加独立完整链路统计，不替换已有 reward 阈值指标。
- `scripts/eval_checkpoint.py`：版本选择、固定 checker、同 trace 双评分、可选 trace 保存。
- `configs/train_config.yaml`：`training.reward_version: v1`，`evaluation.complete_chain_enabled: false`。
- 新增 `tests/test_reward_v2.py`、`tests/test_phase3_integration.py`。

没有修改 `src/tools.py`、`src/scenarios.py` 或任何 JSONL；没有改 ART/vLLM/Unsloth 内部代码或引入依赖/API/Judge。

## 2. 为什么修改

step 2 确定性日志：50 条轨迹、平均工具数 0.94、no-tool 6%，计数唯一对应 47 次调用和 47 条有工具轨迹，即所有非空轨迹恰好调用一次工具。

但 3 个 `country_compare` 场景（预期两次 lookup）都被旧 `reward > 2` 记为成功。只调用第二个国家的 lookup，旧评分就可能达到 2.5。这是相对完整工具链的假阳性，不是真正两步执行能力。

另一方面，数学任务的 `441+23` 与 `441 + 23` 在 v1 中不同；动态 value 与 `__dynamic__` 也不能 literal compare。评分器变化会改变指标，因此不能把换 scorer 后的数字提升当成模型学习改善。

## 3. Reward V2

`compute_reward_v2(...)` 返回不可变 `RewardBreakdown`；`compute_reward_by_version(...)` 对两个版本都返回 float。

| 分项 | 满分 | 含义 |
|---|---:|---|
| baseline | -3 | 保持整体 reward 量级 |
| format | 1 | 已知工具名、arguments 为字典的合理调用 |
| tool | 1 | multiset overlap / 最大调用数，不去重 |
| sequence | 1.5 | 50% prefix ratio + 25% exact sequence + 25% count ratio |
| argument keys | 1 | 同位置同工具的 key Jaccard，按最大调用数归一化 |
| static + dynamic values | 合计 1.5 | 按全部 GT 参数数归一化，缺失参数不忽略 |
| completion | 1 | 次数、顺序、参数和可用执行证据完整 |

- 无调用仍是 -3；正确完整轨迹最高 +4。
- 正确前缀、部分重复调用得到增量 credit，错误顺序和多余调用降低奖励。
- 未完成链路的总分额外限制为 **<=2**，防止再次用旧 `>2` 阈值把部分链路算成完成；`cap_penalty` 单独列出，全部贡献可复算总分。
- 总范围仍是 [-3, +4]，不改变学习率、GRPO 或 group membership。

### 参数 normalization

- `15`、`15.0`、`"15"` 数值等价；bool 不被当作 0/1，非有限数不能获取 dynamic credit。
- 字符串大小写与外部空格归一化，支持 KG/kg 等本来就被工具规范化的形式。
- calculator 使用受限数字表达式 AST 进行格式比较；不执行表达式。空格/多余括号可归一化，`441+23` 与 `23+441` 不自动视为等价；`44 1+23` 不会被错误拼成有效表达式。
- 归一化只改变评分，不改变发给工具的参数。真实工具不接受字符串数值时，错误 observation 会阻止 completion 和 >2 奖励。

### Dynamic credit：保守支持当前路由

实际 rollout 的旁路 trace 每条记录为 `{name, arguments, result, turn}`，其中 turn 是生成这次 action 的 assistant 轮次；不加入 ART messages，不替换原始 Choice，不增加/删除 tool message。

- weather -> convert：只匹配前一步 `temperature`，不拿 humidity 混充；单位必须一致。
- country lookup -> calculator：必须使用前一步 `population / area_km2` 的两个数，顺序与除法必须正确；单纯在 observation 里碰到任意数字不能得分。
- calculator/convert -> convert：value 必须与前一步有限数值结果一致。
- 不使用当前或未来 observation；完整 trace 必须绑定到对应实际调用。
- 同一个 assistant Choice 同时生成 weather 和 convert 时，即使数值碰巧正确，也不能获取 dynamic credit：生成 convert 时还看不到 weather observation，consumer turn 必须严格晚于 producer turn。两个互不依赖的 weather 调用允许同轮执行。
- 不支持的 dynamic 参数路由不给 credit。当前不做浮点近似容差、不做 symbolic algebra。

仅给 predicted/GT（没有 trace）的静态单元评分可以获得完整静态分；dynamic credit 需要前一步结果。真正 rollout v2 总是传带 turn 的完整执行 trace。独立完整链路 checker 则无论参数是否静态，都要求完整 trace。独立调用者可提供无 turn 的有序 trace/result-only observation；这是调用者声明的顺序证据，不证明模型内部因果依赖，不能替代实际 rollout 审计。

## 4. 新增独立成功指标

固定 checker 名称：**complete-tool-chain-v1**。对 V1/V2 训练模型使用同一个判定：

1. 调用次数完全一致，重复调用分别计数；
2. 工具顺序一致；
3. 所有必需参数经同一 normalization/dynamic checker 匹配；
4. 无未知多余参数（weather 显式默认 celsius 是例外）；
5. 每次执行都有绑定到调用的非错误结果；infra failed 不能算成功。

它不是最终答案正确性，也不是搜索 query 的语义等价 judge。旧的 reward 阈值字段保留，新增字段：

- deterministic：`complete_chain_accuracy_pct`；
- sample：`complete_chain_pass_at_k_pct`；
- 详细分母、按深度/类型：`complete_chain`。

默认不开启，避免改变既有评估 artifact。使用 `--complete-chain` 开启；可用 `--no-complete-chain` 覆盖 YAML 关闭。

### 同一 trace 双评分

`--compare-rewards` 不生成第二批模型回复、不重新执行工具，直接对同一份实际 trace 计算 v1/v2，并记录 v2 breakdown。基础设施失败/缺失 trace 不进入这份旁路比较的均值，明确记录 excluded 数量；原有 all-attempt reward/成功率分母不变，训练也没有过滤样本。JSON 的 `reward_version` 明确表示本次 rollout 的评分版本，不自动推断 checkpoint 训练时的奖励版本；在 step 2 权重上切到 v2 评分不意味着该模型已经经过 v2 训练。

`--save-traces` 在每个场景的 `attempts` 中保存实际 arguments/result 和 expected_calls，便于核对数学参数、缺步或 dynamic 使用。

trace 按返回 trajectory 对象身份绑定，而非异步完成顺序/奖励排序，防止 ART 在组内重排成员时错配。ART 必须保留返回轨迹对象；若返回替代/缺失/重复对象，评估 fail-fast，而不是伪造对齐。

## 5. 新增测试与服务器验收

新增用例覆盖：v1 frozen golden outputs/别名 identity/忽略 observations；duplicate/prefix/order；dynamic 数值与单位、未来/misaligned observation、两个 density operands；参数 normalization；工具错误与不完整阈值 cap；breakdown 总分与范围；原始 Choices/logprobs/ART messages；版本配置、训练/validation 同步选择；完整链路分母与 infra；同 trace 比较且只生成一次；trajectory 重排；默认功能关闭的回退路径。

先在服务器运行全套：

```bash
python -m pytest tests/ -v
```

通过后，保留原 step 2 checkpoint，不训练，执行一次双模式基线审计：

```bash
python scripts/eval_checkpoint.py \
  --scenarios data/scenarios_val.jsonl \
  --mode both --generations 8 \
  --reward-version v1 \
  --complete-chain --compare-rewards --save-traces \
  --output eval_step2_reward_audit.json
```

再验证真正 v2 rollout 与 ART metrics 集成，不更新模型：

```bash
python scripts/eval_checkpoint.py \
  --scenarios data/scenarios_val.jsonl \
  --mode deterministic --reward-version v2 \
  --complete-chain --compare-rewards --save-traces \
  --output eval_step2_v2_deterministic.json
```

验收重点：

- V1 主 reward 字段仍是原评分；同 trace 的 v1 comparison 应一致。
- 同 trace v2 对正确空格变化给 normalization credit；未完成的 country_compare 不超过 2，完整链路判定为 false。
- 查看实际 trace 后才能断言数学失败确实来自空格，不能只凭 reward=2 猜测。
- V1/V2 两种运行均可显示独立 complete-chain 指标；comparison 不是模型能力提升实验。
- 50 个场景的 deterministic / sample 分母仍是 50 / 400，4 场景/批不变。

训练 opt-in 已提供 `--reward-version v2` 或 YAML `training.reward_version: v2`，但**本阶段不启动训练**。先验收本阶段，下一阶段隔离 infra failure 后再安排受控训练对照。

## 6. 尚未解决

- 训练中的基础设施失败过滤、empty choices 与模型输入错误的精确归因仍是 Phase 4；此次完全保留旧 failure 行为。
- 零方差组仍仅诊断，不自动重采样或改算法；cap 也可能令某些相同错误组无方差。
- best checkpoint / early stopping 留到 Phase 5。
- checker 是严格预期工具协议，不接受所有语义等价的 query/表达式/独立调用重排；不能替代最终 task correctness。例如天气工具本来支持直接返回 Fahrenheit，某些单步解法可能回答正确，却没有完成 GT 的 weather->convert 两步协议；此次不删除该 Tool API 功能、不改 GT，也不把协议失败宣称为最终答案错误。
- 现有数据限制不修改：Guido 的出生年未出现在当前 search 知识库中；density 模板的 km->miles 数值转换不等同正确的平方单位密度转换。完成旧 prescribed chain 不能证明任务语义正确。
- 新增 scoring 能否改善训练、减少单步化仍需后续固定模型/数据/顺序/参数的 V1/V2 对照。没有编造实验表。

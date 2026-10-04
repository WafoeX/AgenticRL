# RL-Trained Tool-Use Agent：项目学习、代码导读与面试应答指南

> 本文是对项目源码的中文梳理，帮助你从“能运行”逐步提升到“能解释设计、定位问题、回答面试追问”。项目原有文档保留不动；`README_zh.md`、`RUNBOOK_zh.md`、`RUNPOD_zh.md` 和计划文档的 `_zh` 版本是中文翻译。

---

## 一、先用一句话讲清项目

这是一个**面向工具调用的智能体强化学习实验**：给一个 Qwen 2.5 0.5B Instruct 模型提供 5 个确定性工具，模型根据自然语言任务生成工具调用，环境执行工具并返回结果，奖励函数比较实际调用和标准调用，最后用 GRPO 根据同一任务下多条 rollout 的相对奖励更新模型。

它不是一个通用聊天系统，也不是生产级插件平台。它的价值在于把一个完整的 agent RL 闭环拆成了可实验、可复现、可测试的最小项目：

```text
场景数据
   ↓
模型生成工具调用
   ↓
确定性工具环境执行
   ↓
记录 Choice、tool result、logprobs
   ↓
奖励函数比较实际轨迹与标准轨迹
   ↓
GRPO 按组计算相对优势并更新 LoRA
   ↓
检查点 / 评估 / 导出
```

---

## 二、项目的核心问题

模型需要学会的不只是“回答答案”，而是：

1. 判断是否需要工具。
2. 选择正确的工具。
3. 生成正确的参数键和值。
4. 在工具返回后继续下一步调用。
5. 在完成工具链后停止并回答。

例如任务“Einstein 在 2024 年多大”：

```text
用户问题
  → search(query="Einstein birth year")
  → 工具返回 1879
  → calculator(expression="2024-1879")
  → 得到 145
  → 最终回答
```

项目把这个过程变成一条可以计算奖励的轨迹，而不是只用最终自然语言答案做模糊评估。

---

## 三、目录职责总览

### 1. `src/tools.py`：环境和工具执行层

这里是训练世界的“物理规则”：

- `WEATHER_DB`、`SEARCH_DB`、`COUNTRIES_DB`：确定性知识库。
- `UNIT_CONVERSIONS`：单位对到转换函数的映射。
- `TOOL_SCHEMAS`：发给模型的 OpenAI function calling schema。
- `_safe_calc`、`_get_weather`、`_search`、`_convert_units`、`_lookup_db`：真正执行工具的函数。
- `_EXECUTORS`：按工具名路由执行器。
- `execute_tool`：对外统一入口。

关键设计是**没有外部网络请求**。同一输入得到同一输出，训练奖励不会因为天气 API 或搜索 API 的变化而漂移。

### 2. `src/scenarios.py`：任务和标准轨迹生成

`SCENARIO_TEMPLATES` 描述任务模板。每个模板主要包含：

- `id`：场景类型。
- `task`：自然语言任务模板。
- `params`：用随机数生成具体参数。
- `expected_tools`：预期工具名。
- `expected_calls`：用于描述调用形状的字段。
- `build_calls`：根据具体参数构造标准工具调用。

`generate_scenario(seed)` 使用局部 `random.Random(seed)`，不污染全局随机状态；`generate_dataset` 通过 `seed+i` 生成稳定数据。训练集 seed 从 0 开始，验证集 seed 从 10000 开始，降低重复风险。

### 3. `src/rewards.py`：可学习的反馈信号

奖励不是简单的 0/1：

```text
R_format = 1，如果至少调用了一个工具，否则 0
R_correct ∈ [-3, 3]
R_total = R_format + R_correct
R_total ∈ [-3, 4]
```

`R_correct` 由三部分组成：

1. 工具名称集合的 Jaccard 相似度。
2. 匹配工具的参数键 Jaccard 相似度。
3. 参数值字符串精确匹配比例。

这样“调用了工具但参数错了”与“完全没有调用工具”得到不同信号，比单纯成功/失败更容易训练。

注意：工具名比较使用集合，不保留调用顺序；同名工具的参数匹配按第一个未使用的预测调用配对。这是当前实现的简化，不是严格的序列编辑距离。

### 4. `src/rollout.py`：模型—工具交互循环

一次 rollout 的数据流：

1. 初始化 system 和 user 消息。
2. 创建 `AsyncOpenAI` 客户端，指向 ART 提供的推理服务。
3. 调用 chat completion，并传入工具 schema。
4. 将 assistant 的原始 `Choice` 保存到 `messages_and_choices`。
5. 解析 `tool_calls`，执行工具。
6. 把 tool 消息追加到上下文。
7. 循环直到文本回答、空 choices 或达到 `MAX_TURNS=6`。
8. 调用 `compute_reward`，返回 `art.Trajectory`。

这里有一个面试必问的细节：

- 下一次 OpenAI 请求需要普通字典，所以 `_build_api_messages` 会把 Choice 转换为字典。
- 但训练数据必须保留原始 Choice，因为 ART 要从 `choice.logprobs` 构造训练张量。
- 因此“用于请求的消息格式”和“用于训练的轨迹格式”必须同时维护。

每个请求都传 `logprobs=True` 和 `top_logprobs=1`。如果没有 logprobs，ART 可能打印 `SKIP: all rewards equal`，即使你看到组内奖励统计有变化。

### 5. `src/train.py`：GRPO 训练编排

训练过程可以拆成四层：

1. **数据层：** 读取训练和验证 JSONL。
2. **分组层：** `rollouts_per_step / num_generations` 个场景，每个场景生成 `num_generations` 条 rollout。
3. **诊断层：** 打印平均奖励、平均工具数、每个 group 的 mean/std/min/max。
4. **更新层：** `backend.train(model, train_groups, learning_rate=...)`。

GRPO 的关键不是只生成一条答案，而是针对同一个 prompt 生成一组答案，用组内相对奖励计算优势。因此：

```text
rollouts_per_step 必须能被 num_generations 整除
每个 group 必须包含同一场景的多条 rollout
至少一些 group 要有非零奖励方差
```

该文件还给 `LocalBackend._monitor_openai_server` 加了异常吞掉型包装器，规避 ART 后台监控与 LoRA 注册之间的竞态。它是兼容性补丁，不是 GRPO 算法本身。

### 6. `src/evaluate.py`：离线评估

评估使用与训练相同的工具环境和奖励函数，但：

- `temperature=0.0`，减少随机性。
- 不执行梯度更新。
- 对每个场景记录 reward、工具序列、轮数和 success。
- 当提供 baseline URL 时，比较 RL 模型与提示模型的准确率。

`EvalReport.accuracy` 定义为 reward > 2.0 的比例，而不是“最终回答字符串是否完全相同”。

### 7. `scripts/`：运行时脚本

- `generate_scenarios.py`：生成训练/验证 JSONL。
- `patch_art_tokenize.py`：修改安装在 site-packages 中的 ART 文件，使 transformers 5.x 返回的 `BatchEncoding` 归一化为 token list。
- `eval_checkpoint.py`：直接加载 ART 最新检查点，按场景运行多次 rollout。
- `demo_comparison.py`：为视频准备几个固定任务，对比 0.5B RL 模型与 7B 提示模型。
- `export_model.py`：可选合并 LoRA，再推送到 HuggingFace。

---

## 四、从用户请求到模型更新：完整时序

以一条训练场景为例：

### 第 1 步：准备场景

```json
{
  "task": "Convert 98 kg to lbs.",
  "expected_calls": [
    {"name": "convert_units", "arguments": {"value": 98, "from_unit": "kg", "to_unit": "lbs"}}
  ]
}
```

### 第 2 步：模型请求工具

模型通过 OpenAI 兼容接口返回：

```text
assistant.tool_calls[0].function.name = "convert_units"
assistant.tool_calls[0].function.arguments = "{...}"
```

同时返回的 Choice 中包含 token logprobs。

### 第 3 步：环境执行

`execute_tool` 找到 `_convert_units`，返回确定的字符串结果，例如 `215.??`。

### 第 4 步：继续上下文

工具消息带着 `tool_call_id` 回填给模型。模型可以停止，也可以继续调用工具；循环最多 6 轮。

### 第 5 步：奖励

`parse_tool_calls_from_messages` 是通用解析器；训练 rollout 中实际调用列表已经由循环维护，最终交给 `compute_reward`。正确工具、参数键和值会产生高奖励。

### 第 6 步：组内比较

同一场景生成 8 条 rollout：

```text
[-3.0, -0.5, 4.0, 1.2, 4.0, -1.0, 0.5, 2.0]
```

GRPO 使用这些相对差异，而不是依赖一个绝对标注损失，更新模型的 LoRA 参数。

---

## 五、建议的学习路线

### 第 0 阶段：先跑不依赖 GPU 的测试

先读并运行：

```bash
python -m pytest tests/ -q
```

建议顺序：

1. `tests/test_tools.py`
2. `tests/test_rewards.py`
3. `tests/test_scenarios.py`
4. `tests/test_rollout.py`

这一阶段重点理解纯 Python 数据流，不要一开始就被 vLLM、CUDA 和 ART 的依赖问题分散注意力。

### 第 1 阶段：只学工具环境

阅读顺序：

1. `TOOL_SCHEMAS`
2. `_EXECUTORS`
3. `execute_tool`
4. 每个工具的测试

练习：增加一个确定性工具，例如 `get_time_zone`，同时新增 schema、执行器和测试。不要直接改训练逻辑。

### 第 2 阶段：理解奖励函数

手算以下例子：

- 完全没有调用工具。
- 工具名正确但参数值错误。
- 工具名正确、参数键正确、值只错一个。
- 多步调用缺少第二步。

然后检查 `compute_reward` 的中间量：`name_score`、`param_key_score`、`param_val_score`、`raw`。

### 第 3 阶段：理解场景和轨迹

重点看一个单步模板、一个两步模板和一个三步模板，追踪：

```text
seed → params → task → build_calls → expected_calls
```

再阅读 `rollout.py`，画出消息列表在每一轮如何增长。

### 第 4 阶段：理解 GRPO 的 batch 形状

先不启动 GPU，回答清楚：

```text
rollouts_per_step = 32
num_generations = 8
scenarios_per_step = 4
```

这意味着每步有 4 个 prompt group，每个 group 有同一场景的 8 条轨迹，总计 32 条 rollout。

### 第 5 阶段：最后再处理基础设施

按 `RUNPOD_zh.md` 或 `RUNBOOK_zh.md` 配置：

1. torch 2.8+
2. ART `[backend]` extra
3. vLLM `0.17.0+art1`
4. tokenize 补丁
5. 32 rollout smoke test
6. 完整训练
7. 评估和导出

不要跳过 smoke test。它的意义不是证明模型效果，而是证明依赖、推理服务、工具调用、logprobs、trajectory 和 GRPO 更新能连成闭环。

---

## 六、代码中的关键设计取舍

### 1. 为什么工具必须确定性

强化学习需要把 reward 变化归因到模型行为。如果天气或搜索结果随时间变化，模型得到的差异可能来自外部 API，而不是调用质量。内置知识库牺牲了真实性，但换来了可复现性。

### 2. 为什么奖励要拆分

只有“最终答对/答错”会造成稀疏奖励。拆成格式、工具名、参数键和值后，即使模型只选对了工具，也能得到比完全不调用更有意义的反馈。

### 3. 为什么需要温度和 top-p

GRPO 需要同一组 rollout 之间有奖励差异。如果同一个 prompt 每次都生成相同调用，组内标准差为 0，优势信号消失。这里用 `temperature=1.2`、`top_p=0.95` 增加探索。

### 4. 为什么保留 Choice 而不是只保存 JSON

普通 JSON 足以重放对话，却丢失生成阶段的 token logprobs。ART 的训练张量需要这些 logprobs，所以轨迹保留原始对象，API 请求前再临时转换。

### 5. 为什么要做 monkey patch

`_monitor_openai_server` 是 ART 的后台健康检查。已知竞态发生在 LoRA 注册和检查请求之间；项目选择局部包裹异常，让监控失效时训练仍能继续，而不是修改 ART 安装包的核心训练逻辑。

---

## 七、必须主动说出的局限和风险

面试时不要把实验项目包装成生产系统，主动讲清边界反而更可信：

1. **数据规模小：** 200 条训练场景、50 条验证场景、训练只完成了计划 50 步中的 16 步。
2. **工具是模拟的：** 没有真实 API 延迟、权限、限流、脏数据和服务不可用问题。
3. **奖励不是严格序列匹配：** 工具名用集合 Jaccard，不能完整表达调用顺序。
4. **动态参数占位符：** 某些场景用 `__dynamic__` 表示上一工具返回值；当前奖励逻辑是字符串/值匹配，动态值处理是实验性简化点。
5. **`_safe_calc` 不是生产安全沙箱：** 字符白名单和禁用 builtins 降低了风险，但没有资源限额、AST 级执行策略或进程隔离。
6. **运行环境脆弱：** ART、transformers、TRL、vLLM 之间有严格版本耦合，需要定制 wheel 和补丁。
7. **监控补丁会降低可观测性：** `_safe_monitor` 只打印异常并抑制传播，真正的 vLLM 故障仍需额外健康检查。
8. **评估集不等于泛化能力：** 训练和验证模板来自同一确定性模板族，不能证明模型对真实开放域工具任务泛化。
9. **当前结果不代表“0.5B 超过 7B”：** README 中的 demo 是实验对比，不能从小样本结果推出普遍结论。

---

## 八、面试高频拷打与推荐回答

### Q1：为什么用 GRPO，不用 PPO？

**回答：** 这个项目的任务天然能为同一个 prompt 生成多条候选轨迹，并直接比较组内 reward。GRPO 不需要单独训练 value model，而是用组内相对奖励构造优势，工程上更适合这个小型工具调用实验。代价是必须保证组内有足够的奖励方差。

### Q2：GRPO 在本项目中一批数据的形状是什么？

**回答：** 以 `rollouts_per_step=96`、`num_generations=8` 为例，每步 12 个场景 group，每个 group 是同一场景的 8 条轨迹，总计 96 条。训练脚本会打印每组 mean/std；如果所有 std 都为 0，基本没有有效的相对学习信号。

### Q3：为什么要传 `logprobs=True`？

**回答：** reward 只能告诉我们轨迹好不好，策略梯度还需要生成 token 的对数概率。ART 从原始 Choice 的 logprobs 组装训练张量；缺失时 `_get_packed_tensors()` 可能返回 `None`，训练会以误导性的 “all rewards equal” 跳过。

### Q4：为什么不能把 Choice 转成字典后保存？

**回答：** 字典适合再次发送给 OpenAI API，但会丢失 Choice 上的 logprobs。正确做法是轨迹保存原始 Choice，下一轮请求前通过 `_choice_to_message_dict` 临时转换。

### Q5：奖励为什么不是简单的 0/1？

**回答：** 工具调用是结构化、多步行为。格式分鼓励模型先调用工具，工具名分鼓励选对工具，参数键和值分鼓励调用精确。分解奖励能减少稀疏奖励，但当前实现仍是简化的 Jaccard 和精确匹配。

### Q6：Jaccard 的缺点是什么？

**回答：** 集合不表达顺序，也不表达重复次数。例如 `[lookup_db, calculator]` 和 `[calculator, lookup_db]` 的工具名集合相同；两个相同工具的多次调用也需要额外配对逻辑。因此更严谨的方案可以使用序列编辑距离、带位置权重的匹配，或者按步骤分别评分。

### Q7：如果工具返回值会影响下一步参数，当前代码如何处理？

**回答：** rollout 会把工具结果写回消息，所以模型可以读取结果并生成下一步参数。场景模板用 `__dynamic__` 表示动态值，但当前 ground truth 和 reward 处理并没有把所有动态占位符规范化为语义等价比较，这是已知的实验简化和改进方向。

### Q8：为什么工具环境不用真实 API？

**回答：** 这是为了让训练奖励可复现、测试稳定、调试可归因。生产系统应另外增加真实 API adapter、超时、重试、限流、鉴权、缓存和故障注入测试。

### Q9：`_safe_calc` 用了 `eval`，安全吗？

**回答：** 它先限制字符集合并禁用 `__builtins__`，对本项目的固定训练输入降低了代码注入风险。但它不应被当作不可信代码执行沙箱：仍缺乏资源限制和进程隔离。生产实现应使用 AST 白名单解释器或独立沙箱。

### Q10：为什么 temperature=1.2，而评估是 0？

**回答：** 训练需要探索和组内差异，所以使用较高 temperature 与 top-p；评估希望降低随机噪声，使用 temperature=0。两者目标不同。

### Q11：`Already borrowed` 是什么，怎么处理？

**回答：** 是特定 vLLM 定制版本在高并发 guided tool-calling 下的并发锁竞争错误。rollout 只针对这类错误做指数退避重试，超过次数后返回失败轨迹；不能无条件重试所有 `BadRequestError`，否则会掩盖真实请求错误。

### Q12：为什么训练还要 monkey-patch ART？

**回答：** ART 后台 `_monitor_openai_server` 在 LoRA 注册竞态窗口中可能收到 404，并将异常传播导致 proxy 关闭。项目暂时包裹该监控函数，保证监控异常不杀死训练；这不是永久替代上游修复的方案。

### Q13：如何判断训练是否真的在学习？

**回答：** 不能只看程序退出码。要看：组内奖励方差是否存在、`loss` 是否出现、平均奖励趋势、工具调用数量、每 5 步验证奖励，以及独立验证场景上的成功比例。还要比较不同随机 seed 和未训练基线。

### Q14：如何防止数据泄漏？

**回答：** 训练和验证使用不同 seed 起点，但来自相同模板族，仍可能存在模板层面的分布相似。更严格的实验应按模板类型、实体、参数组合做隔离，并增加手工设计的隐藏测试集。

### Q15：如果让你继续改进，你会做什么？

**推荐分三层回答：**

- **奖励层：** 动态参数语义匹配、顺序感知奖励、工具结果正确性、对重复/无效调用惩罚。
- **数据层：** 扩大模板和实体，分离模板级验证集，加入困难负例与工具失败场景。
- **系统层：** 固定 lockfile，移除对 site-packages 的直接 patch，增加服务健康检查、断点恢复、实验追踪和多 seed 统计。

---

## 九、现场白板题：如何自己推导一次奖励

假设标准调用是：

```python
[
    {"name": "calculator", "arguments": {"expression": "3+5"}}
]
```

预测调用也是完全一样：

- `r_format = 1`
- 工具名 Jaccard = 1
- 参数键 Jaccard = 1
- 参数值匹配比例 = 1
- `raw = 1`
- `r_correct = 6 * 1 - 3 = 3`
- `total = 1 + 3 = 4`

没有任何预测：

- `r_format = 0`
- 函数立即返回 `0 - 3 = -3`

工具名正确但表达式错误：

- 格式分仍为 1。
- 工具名和参数键仍可能匹配。
- 参数值匹配为 0。
- 奖励介于完全正确和完全错误之间。

这道题能证明你真正读过奖励实现，而不是只会复述 README。

---

## 十、最后的面试自我介绍模板

> 我做的是一个小型工具调用 Agent RL 实验。模型是 Qwen 2.5 0.5B，训练框架是 OpenPipe ART，优化方法是 GRPO。环境提供五个确定性工具，场景生成器从模板生成单步到三步任务；rollout 保存原始 Choice 和 logprobs，工具返回结果继续回填上下文；奖励拆成格式、工具名、参数键和值三部分。训练时同一个场景采样多条轨迹，利用组内相对奖励更新 LoRA。项目最大的工程难点不是模型代码，而是 ART、TRL、transformers 和定制 vLLM 的版本耦合，以及高并发下的 vLLM 锁竞争和 ART 后台监控竞态。这个项目的局限是数据和训练步数较小、工具是模拟的、奖励对动态参数和调用顺序仍然简化；如果继续做，我会优先改进顺序感知奖励、动态结果匹配和模板级验证集。

如果你能用自己的话讲完上面这段，并能现场解释一条轨迹如何变成 reward 和 GRPO group，就已经从“会运行项目”进入“理解项目设计”的水平。

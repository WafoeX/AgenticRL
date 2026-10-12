# R1.6 / R2.1–R2.4：独立答案 checker 与配对干预 CPU 验收

> 本轮完成研究开发代码与 CPU 验收，不是模型能力实验。没有模型推理、GPU、SFT/GRPO、commit/push；没有冻结 test。旧工具/数据及整个 `data/research_v1_candidate/` 保持不变。

## 1. 完成范围与文件

- `src/research_answers.py`：纯 stdlib 独立 oracle、严格最终 JSON 解析、固定容差。
- `src/research_checker.py`：答案/真实执行/严格链路/语义路线/动态依赖分开返回；规范化 chat 的终止消息适配。
- `src/research_interventions.py`：开发配对快照、响应诊断和保留计划组分母的汇总。
- `scripts/diagnose_research.py`：只读候选 validation，CPU scripted 正/负控，原始输出、manifest 和字节重生成检查。
- `tests/test_research_checker.py`：95 项新增测试；覆盖全部 16 recipe 的新建开发 fixtures、手算转换、合法替代、对抗 trace、快照隔离和 CLI 拒绝覆盖。

对应任务书 **R1.6、R2.1–R2.4**。**R2.5–R2.8 未完成**：没有实现/验证 Base 与指定 checkpoint 的实际加载身份、固定模型解码配置、正式推理尝试/服务事故处理、正式分层模型评测入口。CPU 诊断留档不是这些 GPU/后端项目的替代验收。R0.3 正式预注册与 R1.8 test 冻结仍未完成。

## 2. R2.1：答案解析与固定规则

版本：`research-answer-v1`。

统一最后一个终止 assistant 回复只输出一个 JSON object，**且不含 tool calls**：

```json
{"value": 68.0, "unit": "fahrenheit"}
```

- 恰好 `value` / `unit` 两个键。允许首尾空白和键顺序变化；不抽取自然语言/代码围栏中的局部数字，不修复 JSON，不挑多个答案中的正确项。
- 拒绝重复键、额外 `success`、数组、布尔数值、数值字符串、null、NaN/Infinity、超浮点可表示范围数字；最终回复上限 16 KiB。
- 单位词表与固定 system prompt 一致；不自动换单位，不接受单位别名、大小写变体或缩写。
- 比较答案是精确实体字符串或 `tie`，不把 A/B、实体大小写改写、`Tie` 自动视为同义答案。`condition` 字符串解析已支持，但当前数据的 `simple_weather` 只问温度；不能声称新增了未注册的天气 condition/humidity oracle。
- 当前整数/离散 recipe（算术、年份差、GDP及GDP缩放）数值精确；比较字符串精确。其余 number 使用 `abs_tol=1e-6`、`rel_tol=1e-9`，判定 `abs(pred-gold) <= max(abs_tol, rel_tol*abs(gold))`。
- 容差由 checker 固定，不读取模型或可改的 `record.answer_spec`。转换采用独立 Decimal、6 位 half-even；面积密度倍率是平方面积，手算例 `1000/10 * 1.609344² = 258.998811`。
- `final_content` / `check_messages` 只检查**最后一条**规范化消息：尾部是 tool、空内容或 assistant tool call 时记 missing output，不回退到较早的正确 JSON。真实 ART/OpenAI call ID / turn 来源适配尚属后续评测入口工作。
- exception / truncated / infra_failure / missing_output 分开记录；终止不完整不计答案成功。不把一次基础设施失败默默删除。

## 3. R2.2：oracle 的独立性与共同错误边界

`oracle_answer(record)` 只使用 `recipe_id`、公开 `parameters`、原始 `environment_snapshot`。不读取保存的 `oracle_answer` / `reference_answer` / `reference_trace` / `expected_calls` / `useful` / reward / 模型 success。

答案模块只导入 `json` / `math` / `decimal`；没有导入生成器、工具环境或旧 V1/V2 reward。转换使用独立 SI 定义和 Decimal 计算，而不是调用工具转换或生成器内部 gold 公式。16 recipe 全部显式注册，未知 recipe fail closed。对毒化保存 gold/参考/容差、以及故意破坏工具 executor 的测试，独立答案判定不变；真正改变原始事实则 oracle 改变。

**边界：**checker 与环境仍共享结构化事实及 recipe 任务定义。若任务文字、原始快照或整个语义约定共同写错，独立计算不能自动纠正它们。手算例、原始事实审计用于交叉验证，不把该 oracle 宣称成外部盲测或数学形式化证明。

## 4. R2.3–R2.4：答案、链路、语义与依赖分开

版本：`research-trace-v1`。

每个可信 flat trace event 具有 `turn / name / arguments / observation`。按不可变快照重放工具，并与保存 observation 精确比较；真实工具异常与虚构 observation 分开分类。`turn` 是产生调用的 assistant turn，不能给同一批依赖调用伪造逐工具递增的 turn。

独立构建协议路线，**不信任数据保存的参考调用或 dependency graph**：

| 输出 | 含义 |
|---|---|
| `answer.correct` | 独立最终答案主判定；不因缺步、无工具、reward 高低或非参考路线而改判 |
| `execution_valid` | 每个 observation 与执行器重放一致，参数/工具没有执行错误 |
| `strict_complete` | 注册路线完整、动态时序有效、无额外工具；不检查最终答案是否正确 |
| `semantic_valid` | 有可核对的完整注册语义路线或注册合法替代；允许额外真实工具调用，但不容忍虚构结果 |
| `dynamic_status` | `valid` / `invalid` / `not_applicable` / `alternative_bypass` |

两比较查询可反序或同 turn 执行。算术接受括号/数值拼写变化及 `+`、`*` 交换顺序；不把相同运算结果当成相同语义，如 `2+2` 与 `2*2` 不同。未注册的任意代数等价路线可能不获语义/严格链路信用，**但正确最终答案仍获答案信用**。无工具的正确答案也获答案信用，但没有工具路线/因果依赖证据。

天气直接查询 Fahrenheit 为合法一调用替代：答案正确、语义路线合法，严格参考两步未完成，动态边记 `alternative_bypass`，不混入动态依赖通过计数。

对抗测试覆盖：缺步但最终正确、错误参数、虚构 observation、同 turn 伪依赖、错误平方单位、答案对但协议不同、协议完成但答案错、异常/截断/infra/缺失终止消息、非法 trace 类型/字段、额外真实/虚构调用。reward 不参与任何答案判定。

**不能证明的事：**可信日志中数值/结构吻合和先后时序不是模型因果注意证明；本 checker 也不认证外部伪造的 turn 或 tool 来源。协议重放共享 `ResearchEnvironment`，共同工具实现错误风险须披露。后续模型入口必须保存原始消息/call ID、真实批次与 observation 来源，不能丢弃 ART 的 token/logprob 信息来迎合本 flat trace 格式。

## 5. R1.6：开发配对干预

每组保留原始 `semantic_group_id` 和 split；base 与 variant **问题文本、实体、公开参数、system prompt 和 tool schema 完全一致**。只改变隐藏 producer 事实，且每个 variant 在整个 rollout 内固定不变：

- birth year：+7；distance：+7 km；temperature：+7 °C；GDP：+7 billion USD。
- population：增加约 7 倍原面积的整数人口，面积保持不变，确保密度与后续转换显著改变。

新对象是**开发诊断 view**，不是修改/重写数据 schema 的正式 record：删除保存的 gold、参考轨迹与旧语义/值签名，保留源组 identity 并重算新环境 hash。不能把它误写回 train/test 或混作额外主准确率样本。

`make_pair` / `build_pairs` 拒绝所有 test 来源，即使这些 test 行不需要动态调用也不接受。两个世界分别构造不可变环境，线程混合执行验证不串扰。每个 consumer 参数签名与独立答案必须都发生变化；天气合法单查询绕过 consumer，则单列 bypass 的 paired response。

汇总按**计划语义组**计数，缺失组不缩小分母。responsive 要求双方答案正确、双方真实语义路线有效、答案改变，且全部动态 consumer 参数改变；合法 Fahrenheit 绕过分开报告。这是受控响应诊断，不宣称模型内部机制的因果识别。

## 6. 实际 CPU 证据

产物：`results/research/checker_interventions_v1/`。

| 开发检查 | 结果 |
|---|---:|
| validation scripted 参考独立答案/严格链路 | 180/180 |
| 动态配对组；consumer 参数与答案随事实改变 | 106/106 |
| 冻结参数（但真实执行 variant 工具）拒绝 | 106/106 |
| adaptive 轨迹但保持旧答案拒绝 | 106/106 |
| 整条旧轨迹虚构为 variant observation 拒绝 | 106/106 |
| variant 同 turn 预填全部依赖调用拒绝 | 106/106 |
| Fahrenheit 一调用合法替代 | 10 组通过；全部 dependency bypass |

106 组来自 validation：age10、density-convert36、GDP-scale12、population-math10、search-convert10、weather-convert10、weather-offset-convert18。另两种 test 专属 recipe 只用**新建人工开发 fixtures**验证代码，未读取正式 test 行给本 checker 调试，也未跑任何模型 test。

这是读取真实 observation 的 **scripted responder 正控**与4种负控，不是 Base/SFT/GRPO 的准确率、Pass@8 或训练增益。不能把106组×2世界当212个独立主指标样本，也不给脚本正控套模型置信区间。

验证记录：

- 新增 **95 tests**；全量 **370 passed in 6.09s**，包含原275项回归。
- fresh standalone Pyright：五个新增 Python 文件 **0 errors / 0 warnings**。
- Ruff `E,F,I,B`（忽略 E501）：通过。
- 诊断 `--check`：完整产物字节一致；既有候选生成器 `--check`：六个原始产物字节一致。
- manifest 包含 source hashes、validation hash、prompt/tool schema hashes、旧数据/工具前后哈希、各产物大小/SHA-256；raw JSONL 保留每组双方答案/trace/checker 和各负控，不只保存汇总。
- 编辑器新文件索引的 missing imports、`.execute` 被误识成 SQL、已验证 Decimal 的 numeric parse 提示已按诊断身份记 false-positive；真实类型和排序问题已修复。没有添加源码 ignore。session disposition 隐藏提示不等于独立 clean 证明，以上 fresh 检查与实际运行才是证据。

诊断脚本只消费 validation。全量回归和原候选 `--check` 会执行既有程序化数据构建/格式/字节校验（包括候选 test），不等于对 test 跑模型或用模型 test 成绩修改协议。

## 7. 复现与下一关卡

```bash
# 目录不存在时生成；已有目录拒绝覆盖
python3 scripts/diagnose_research.py
# 不写文件，完整重生成/字节核对
python3 scripts/diagnose_research.py --check
python3 scripts/generate_research_dataset.py --check
uv run --no-project --with pytest --with pytest-asyncio --with psutil python -m pytest tests -q
```

若要留新版本，显式使用新的 `--output results/research/<new-id>`，不要覆盖旧证据。

下一项：R2.5–R2.8 统一模型推理/指定checkpoint身份、实际解码预算/attempt与分层统计，补齐正式预注册及冻结纪律；之后才能讨论 GPU pilot 和 SFT/GRPO。本轮未自动推进训练或消耗 GPU 预算。

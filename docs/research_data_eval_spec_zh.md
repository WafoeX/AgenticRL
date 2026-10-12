# Research V1：数据与独立评测规格

> 2026-10-11；R1/R2 实现规格。设计选择已获用户确认，**候选数据、R1.6/R2.1–R2.4 checker及R2.5–R2.8本地评测入口CPU合同验收完成；Base CPU pilot已完成加载/生成/重放；R0.3决策预注册与R1.8操作性test冻结已于2026-10-12完成，正确多步/checkpoint/目标后端与预算附录待验**。证据见 [数据验收](research_dataset_implementation_zh.md)、[干预/checker验收](research_checker_interventions_zh.md) 与 [本地评测合同](research_evaluation_zh.md)。
> 约束：Qwen2.5-0.5B、5 类工具、Colab A100 40GB、总计 20 GPU 小时、两周目标。
> 本文不包含虚构实验成绩或未测量的训练耗时；正式训练预算与后端兼容参数仍待 pilot。

## 1. 已确认的选择

2026-10-11 用户问卷确认：

1. 每场景使用**固定种子的合成环境快照**，不是不断变化的实时服务。
2. 所有实验统一**结构化 JSON 最终答案**，不以 LLM judge 作为主评测。
3. 构建目标为 **train 1200 / validation 180 / test 240 个独立语义实例**；test 的同分布、未见参数、未见表达、新组合各 60 个；整体 1/2/3-call 为 30%/40%/30%。

旧模板的问题与证据见 [R1.1 审计](research_dataset_audit_zh.md)。新版本不是旧 200/50 数据的无损扩容，必须重新跑 Base/SFT 等基线，不能直接拼接历史准确率。

## 2. 环境与兼容边界

- 保留 calculator、get_weather、search、convert_units、lookup_db 五类工具；没有联网 API，没有新增 Agent 功能。
- 新工具环境由独立构造器生成 `environment_snapshot`，使用局部 RNG，不依赖进程随机状态、Python 的随机化 `hash()` 或系统时钟。
- 同一实例中每个工具返回固定；不同模型/奖励/采样尝试读取**同一个快照**。生成模型的 sampling seed 不改变环境。
- 新快照采用虚构人物/城市/国家/测量记录，提示明确是 benchmark 合成事实；不能给现实 Paris/France 随意编造事实，再包装成真实世界问答。
- 采用可选上下文接口或独立研究环境对象；未传上下文时旧 `src/tools.py` 行为保持原样。不能用可变模块全局 DB 承载并发场景，防止不同 rollout 的世界互相串扰。
- snapshot/工具版本须哈希并留档；隐藏值仅经 tool observation 返回给模型，不出现在任务/system prompt 的 gold 字段中。
- calculator 保留受限算术；参考轨迹不用未定义符号或任意代码。单位常量固定，密度单位显式支持 people/km²→people/mile²，采用 `1 mile = 1.609344 km`。
- 新环境的公式、舍入、知识检索契约必须写测试。search 对合成 entity/字段有确定检索结果，不依赖旧词重叠算法碰巧命中。

**CPU验收已通过：**同实例固定快照跨顺序/线程返回一致、不同上下文隔离；旧194项测试无回归，本轮81项新增，全量275passed；模型消息白名单不包含oracle/expected_calls。尚未进行GPU/跨模型推理评测。

## 3. 数据记录与隐藏字段边界（R1.2）

以下字段已在 `src/research_schema.py` 与 `configs/research/schema_v1.json` 落地，并验证全部1620条候选记录；保存gold是内部构建信息，R2独立checker另从原始快照重算，不信任该字段：

| 字段 | 意义 | 可发送给模型？ |
|---|---|---|
| `schema_version`, `dataset_version`, `sample_id` | 数据和唯一实例身份 | 不作为答案提示；日志使用 |
| `family`, `recipe_id` | 原子/复合语义任务与组合配方 | 否 |
| `task`, `wording_id` | 自然语言任务和表达模板身份 | 仅发送 task |
| `semantic_group_id`, `semantic_signature` | 去重/分组身份，不能只用 random seed | 否 |
| `split`, `generalization` | train/validation/test 与分层类别 | 否 |
| `parameters` | 类型明确的输入参数、实体、操作/单位 | 只有 task 明确需要的部分 |
| `environment_seed`, `environment_snapshot`, `environment_hash` | 固定本地工具环境 | 仅工具执行器可用，按查询返回 observation |
| `reference_depth`, `dependency_graph` | 参考调用深度及结果消费边 | 否 |
| `expected_calls`, `reference_trace` | 评分/审计用参考轨迹，真实执行生成 | 否 |
| `answer_spec`, `oracle_answer` | 答案类型、单位、容差与 gold | 否；通用输出格式不包含 gold |

`sample_id` 必须全局唯一，但**不等于语义独立**。semantic signature 至少包含语义 recipe、实体/输入、算术操作、单位及所需环境事实；另计算去实体名/ID后的值与操作签名，发现只改实体名的重复。派生结果相同不必然是同一问题，不能仅按 gold answer 去重。

原始参数、任务表达、依赖 graph 均保留，支持反查签名，不只输出 opaque hash。加法/乘法交换等价与独立比较的实体重排需统一规范化；非交换操作不能错误排序。

`expected_calls` 可保留旧训练接口需要的结构，但 `__dynamic__` 仅能作为内部规格标记；真实 expert trace 和模型调用必须是具体参数。动态解析基于已经执行的 observation，而不是直接查隐藏 oracle。

## 4. 任务语义与参考链路

对旧 12 类任务语义修复并扩容：

| 新语义 | 参考工具路径 | 最终答案类型 | 关键约束 |
|---|---|---|---|
| 合成人物年份差 | search→calculator | number / years | 明确目标年份减出生年份，不声称精确周岁；从返回事实填入出生年 |
| 天气温度转换 | get_weather→convert_units | number / fahrenheit | 直接查 Fahrenheit 为合法替代答案，不自动判答案错 |
| 人口密度 | lookup_db→calculator | number / people_per_km2 | 正确消费人口与面积，面积正数 |
| 单步算术 | calculator | number / dimensionless | +、−、×；交换等价去重 |
| 指定天气字段 | get_weather | number / celsius 或 enum | 每任务只明确查询一个字段，避免模糊综合天气 gold |
| 搜索指定事实 | search | number 或 enum / 明确单位 | 返回主字段与精度唯一，不用混合近似单位片段 |
| 搜索测量值后转换 | search→convert_units | number / 指定单位 | 后调用使用真实返回值，非硬编码常数 |
| 密度单位转换 | lookup_db→calculator→convert_units | number / people_per_mile2 | 采用平方单位的反向面积倍率，不能调用 km→miles |
| 两城市比较 | get_weather + get_weather | enum / 实体或 tie | 两查询独立，交换顺序允许；A/B 平衡 |
| 单步单位转换 | convert_units | number / 目标单位 | 参数范围/方向明确，采用固定常量和舍入 |
| 指定 GDP 字段 | lookup_db | number / billion_usd | 合成事实；明确币种与数量级 |
| 两国家人口比较 | lookup_db + lookup_db | enum / 实体或 tie | 两查询独立，交换顺序允许；A/B 平衡 |

调用深度由**参考路径**定义，不是要求所有正确答案都使用相同数量。两次独立查询不标成动态参数依赖；weather_convert 虽有参考依赖路径，却存在合法一调用答案，不能当“必要两步”证据。

组合 holdout 以**语义 recipe / DAG**为单位，不只按工具名列表：

- validation 候选两步 recipe：查 GDP 后乘指定比例；三步 recipe：查天气 → 加指定温度偏移 → 转 Fahrenheit。
- test 候选两步 recipe：检索测量事实 → 乘指定比例；三步 recipe：检索距离 → 加指定公里偏移 → 转 miles。
- 这些 recipe 不进入 train；test recipe 也不进入 validation。原子工具与运算在 train 具备覆盖。
- 两步 recipe 即使共享 search→calculator 顺序，也属于未见**语义组合**，不能宣称“工具序列未见”。三步是否序列未见需要生成后的实际审计。
- 候选配方的参数/值容量、动态 checker 支持和覆盖在实现时校验；变更配方必须先更新规格，在 test 冻结后不能为了成绩修改。

## 5. Split 与计数（R1.3–R1.5）

先生成/规范化独立语义实例并分组，再渲染表达；不得先写 train/val/test 再删除碰撞样本，导致缩小分母或分层失衡。每个实例一个主表达，开发改写需属于同组，不能跨 split 复用。

| Split/分层 | 实例 | 1-call | 2-call | 3-call |
|---|---:|---:|---:|---:|
| train | 1200 | 360 | 480 | 360 |
| val 同分布未见实例 | 90 | 36 | 36 | 18 |
| val 未见参数 | 30 | 9 | 12 | 9 |
| val 未见表达模式 | 30 | 9 | 12 | 9 |
| val 新语义组合 | 30 | 0 | 12 | 18 |
| **val 合计** | **180** | **54** | **72** | **54** |
| test 同分布未见实例 | 60 | 24 | 24 | 12 |
| test 未见参数 | 60 | 24 | 24 | 12 |
| test 未见表达模式 | 60 | 24 | 24 | 12 |
| test 新语义组合 | 60 | 0 | 24 | 36 |
| **test 合计** | **240** | **72** | **96** | **72** |

新组合至少两步，所以**不要求每个泛化分层都为 30/40/30**，只要求每个 split 整体满足用户确认比例。

- 同分布：训练用 recipe/表达风格/取值域相同，语义实例不同。
- 未见参数：对各 family 指定互斥参数域（候选为 train 范围外的数值段）。val OOD 与 test OOD 使用不同段；其余 recipe/表达保持训练分布，尽量隔离变量。字段范围必须合法，不套用一套人口/出生年份的通用数字范围。
- 未见表达：保留 train recipe 与取值域，但使用分开注册的表达模式集合；val holdout 与 test holdout 的 wording ID 不相同。实例仍与 train/val 独立，不能复制同任务只换措辞。
- 新组合：按第 4 节的 reserved recipe；原子工具、参数域、表达难度尽量控制，单列组合造成的分布变化。
- 同时新实体/新参数等因素不混入“单因素泛化”而不披露；初版未承诺单独的新实体主分层，确需增加时单列诊断。

比较任务在相应 recipe 内对非 tie 样本平衡 A/B；tie 在开发/单元测试覆盖，可在正式数据中按明确配额纳入，不能把“不同实体相同值”静默判 A 或 B。家族层面的精确分配由生成器配置公开并验收，总数不得漂移。

去重关卡：跨 split 精确与规范化文本为 0；共享禁止的语义实例组/等价值签名为 0；holdout recipe/wording/参数域没有违约。不得通过给 task 追加 sample ID 获得表面“无重叠”。

## 6. 结构化答案与独立 oracle（R2.1–R2.4）

所有 B0/B1/G1/G2/消融使用相同通用协议：最后一个不含工具调用的 assistant 回复只输出一个 JSON object：

```json
{"value": 68.0, "unit": "fahrenheit"}
```

- `value` 为有限 JSON number 或规范实体/枚举字符串；`unit` 为预注册词表中的字符串。没有布尔数值、NaN/Infinity、数组、多个答案或额外 success 字段。
- 对 enum/实体答案，unit 固定为 `entity` / `condition`；比较相等用 `value="tie"`。单位、实体、tie大小写及拼写精确，不做隐式同义/别名修复。当前天气recipe只问温度；condition解析不等于新增condition任务。
- 允许首尾空白；不从多个 JSON/自然语言中的某个数字择优抽取，不自动把错误单位换成正确后记成功。解析失败与语义错误分开记录，主准确率都计失败。
- `research-answer-v1`固定整数/离散recipe（算术、年份差、GDP/GDP缩放）精确；其他数值abs_tol=1e-6、rel_tol=1e-9，按 `abs(pred-gold) <= max(abs_tol, rel_tol*abs(gold))`。checker不信任保存的可改容差；转换独立Decimal、6位half-even，不设宽容差掩盖平方单位错误。
- oracle 由结构化原始事实和独立公式生成，不读取 V1/V2 reward 或模型自报 success；不可直接照抄 reference trace 的最终 result 作为唯一 gold。
- 密度、单位和年份差用手算样例/独立公式交叉校验，检查参考 trace 与 oracle 一致。共同用同一环境事实并非完全独立，报告披露共同事实输入的边界。
- 单列答案正确、严格链路完成、语义合法替代、动态参数/时序有效性；不要把三个 checker 合成一个 reward>2 阈值。

本协议测量**受控结构化工具任务**，不声称覆盖开放域自然语言答案质量。

**R2.1–R2.4已实现：**`research_answers.py`纯stdlib，不导入工具/生成器/reward；`research_checker.py`分开返回答案、execution/strict/semantic/dynamic结果，只认证可信flat trace的已注册路线。最终回复上限16KiB、拒绝重复键/代码围栏/多答案，终止消息不回退挑答案。合法Fahrenheit绕过记bypass，无工具正确答案照常获答案信用。未注册等价路线可能不获trace信用但不影响答案判定；共享事实/协议重放executor的共同错误风险见验收报告。

## 7. 动态依赖与干预诊断（R1.6）

训练/正式评测先核对：消费参数匹配正确生产字段；生产结果真实可执行；消费 turn 晚于生产 turn。同一 assistant 回复预先生成两个相依调用，不视为已看到本回复内工具结果。

这只是 trace 证据，不能单独证明因果使用 observation。另在开发/validation 诊断用配对快照：保持问题表达、实体与工具 schema 相同，改变隐藏生产事实，检查后续参数与最终答案是否随之改变。

配对实例同组留在同一 split；诊断集按组计算，不混作独立主准确率样本。干预不在同一 rollout 途中悄悄改变世界，不把旧 observation 与新 world 混用。冻结 test 不用于调干预规则。

**R1.6已实现：**`research_interventions.py`只接受train/validation来源的开发view，去除旧gold/参考；固定单个隐藏事实扰动并重算环境hash，公共消息不变。validation106组scripted正控全部参数/答案响应，冻结参数/旧答案/虚构observation/同turn负控各106/106拒绝；10组天气合法绕过单列。这是代码诊断验收，不是模型行为/训练收益。

### R2.5–R2.8：本地评测协议已落地（CPU合同，不是模型成绩）

`configs/research/evaluation_v1.json` 固定 greedy与sample@8（temperature1.2/top_p.95）、seed20261011、6 turns/每turn8 calls、512/2048输出tokens、4096输入tokens、60s请求软限、CPU并发1。精确本地chat template/Qwen文本tool codec，无隐式override/修复，逐turn seed固定；不宣称跨环境字节确定性。

`eval_research.py` 默认preflight，仅显式 `--run-cpu` 加载本地CPU float32模型；Base/LoRA身份、pinned revision、step/Base字节绑定、加载回执、模板/实际config及前后文件hash独立核对。仅train/validation、拒绝test；完整计划先存，原始结果实时落盘，ledger补缺失，不删失败/不补采样。实际返回空模型回复仍计模型失败；空choices/usage合同错误为infra。infra/真正缺失使比较无效。

答案/strict/semantic/dynamic分开，四类分层raw counts及全量/有效分母留档；Pass@8按至少一次成功场景，Wilson场景区间、sample场景cluster bootstrap及配对差值。生产audit从raw响应重放真实工具/provenance/checker，不信任保存success。无数字签名/服务来源认证；未采集训练logprobs，不代替R4 ART训练集成。

CPU fixture两身份各8个validation案例、72计划/72返回/70有效/2infra，比较无效；15文件导出重生成与HASHSEED99通过。100新增tests、全量470passed、fresh九文件Pyright/Ruff通过。真实Qwen/PEFT/ART/vLLM/GPU兼容未验证；脱敏fixture的内层manifest为导出前记录，外层manifest/--check用于导出复核，不能将假权重或模拟百分比当能力证据。详见 [本地评测验收](research_evaluation_zh.md)。

## 8. 奖励与实验公平性

- V1 保留原评分函数作为历史机制 baseline，记录其实现 hash。动态占位符的不足不通过给 V1 独享 gold 参数来补救。
- 现有 V2 的依赖路由不足以支持 search/新 recipe。研究用补充后的 V2 单独标识为 `reward-v2-research-v1`，公布分项、范围、未完成 cap 与路由，正式实验前冻结。
- 研究 oracle 不作为额外训练奖励，除非另行预注册；当前设计只用于监督轨迹校验、选点与独立评测。
- 两个 GRPO 分支共享 SFT 起点、数据/环境/schema/prompt、场景顺序、采样/更新预算、选点评测频次、基础设施失败规则；只有冻结奖励版本不同。
- 旧 V2 与研究 V2 不是同一未修改实现；结论应写“研究版本轨迹奖励与旧 V1 的受控对照”，不能冒称只有一个分项改变。单项消融要基于同一研究 V2。

## 9. 冻结、预注册与预算

实施关卡：研究环境/schema/容量 → split构建与程序化审计/checker → Base开发pilot → 决策预注册及独立test数据冻结（本轮完成） → train-only专家/SFT契约 → SFT加载与GPU成本pilot → 锁定正式训练预算/有效配置附录 → 四组主对照与消融 → 全臂模型身份及最终test就绪锁定 → 单独授权的冻结test批量推理。数据冻结不等于最终模型test解锁。

- 先在人工开发 fixtures 和 train/val 子集调试；不对 test 跑模型、看成绩再改奖励。
- test 的程序化构建/格式/可解性验证允许，但只用预先固定规则，不能根据模型 test 错误回头修正 benchmark 并保留“未见”说法。
- 冻结清单包含数据/hash、生成代码、环境、checker、reward、模型配置、checkpoint 选择、seed、评价参数、planned attempts 和异常处理。
- 主指标为 test 最终答案 deterministic accuracy；分层结果、答案 Pass@8、严格链路、动态依赖、infra 与成本同时报告，不仅平均 reward。
- 240 个 test × (1 deterministic + 8 sample) × 5 个实验臂 = **10800 个计划 test attempts**（单种子）。这不是已执行数量，也不包括 validation、训练或重试。
- validation 选点优先 deterministic，不在每个 checkpoint 都跑 Pass@8；最终正式模型才按固定计划进行完整 sample 评测，以控制20小时预算。
- GPU pilot 测得推理/训练/加载成本后再决定步数与预算分配。若 20 小时内无法覆盖计划，先通知用户缩减或重定协议；不能静默少跑 test 或更改分母。
- 已锁定single-seed20261011；不自动追加种子。validation全180条greedy按答案正确数→reference动态有效数→较早step选点，A1为G2移除动态奖励且不重新缩放。评测无自动重试；GRPO infra整组最多一次重试，仍失败跳过整组（要求尚待R4实现核验）。新增seed/策略变更须事前版本化修订和用户确认。

**候选实现已公开并验收：**family配额、参数域、wording/recipe注册表见 `configs/research/dataset_v1.json` 与候选 `audit.json`；R2答案/trace、干预及本地评测协议版本见源码和各manifest。**本轮已锁定：**[正式决策预注册与操作性冻结](research_preregistration_v1_zh.md)，机器决策JSON、hash锁及`data/research_v1_frozen/freeze.json`；原candidate六文件保持原字节，冻结发布不修改记录内历史candidate标记。**尚待验证/锁定：**正确模型多步、真实checkpoint/ART-vLLM兼容、SFT/GRPO有效配置/奖励hash和实测预算附录；formal_training_ready=false、final_test_ready=false。

## 10. 下一步可执行任务

R1.2–R1.5 第一批及候选R1.7已完成：`data/research_v1_candidate/` train1200/validation180/test240，40个需求池各1024去实体名组，最紧余量613；精确/规范化文本、语义/值/完整轨迹跨split重叠为0；全参考真实执行并与独立公式一致，全部产物字节可重生成。旧 `data/scenarios_*.jsonl` 与工具hash未变。

共享工具事实可能已在train observation暴露，实体新颖性也有交叉因素，两项诊断已单列，不能把分组无泄漏包装成所有事实未见或纯参数OOD。原candidate保留，独立`data/research_v1_frozen/`已操作性冻结；test从未在本轮做模型推理，原构建/可解性审计及标签本地可见性如实披露，不宣称外部盲测。

Base CPU validation pilot已通过真实身份/生成/raw重放，但两场景答案与正确工具链失败。随后本轮R0.3/R1.8完成，26新增冻结纪律测试、全量506passed，freshPyright/Ruff通过；冻结check/HASHSEED99、原候选/干预重生成与既有真实pilot/模拟fixture只读复核通过。下一项单独授权R3.1–R3.3 train-only专家与SFT mask/序列化检查；再做兼容/成本pilot和训练前附录。不自动启动GPU、模型推理或SFT/GRPO。

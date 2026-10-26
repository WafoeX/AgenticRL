# 四模型冻结test：最终结果与反馈复核 v1

## 结论

**在本合成工具使用基准的独立操作性holdout上，答案主导GRPO相对固定SFT30观察到三个训练种子一致的泛化改善。** SFT30答案正确率为60.00%，三seed为75.42% / 71.25% / 70.83%，均值 **72.50% ± 2.53个百分点（样本SD）**；主要比较提升 **12.50个百分点**，同题配对场景bootstrap 95%区间 **[8.33, 16.67]个百分点**。

收益不再仅来自country_density_convert，组合任务也改善；但约98%的净增益仍集中在三个家族，组合任务绝对正确率仍低、参数泛化略退步，且调用/tokens/耗时增加。**不能写成“所有工具能力全面改善”“同时提高效率”或“RL方差低于SFT”。** 本次test至此用于最终结论，不再据此调参、追加训练或挑seed重跑。

## 1. 反馈完整性与协议核查

原始文件：`results/20261018T133835Z-final-test-feedback.zip`，2926853 bytes、977项，SHA256：

```text
cd434cd4e036d613710996817145708eeea7093e16e18ebe9fb9b73c8bef1868
```

- ZIP CRC通过，无重复/额外/缺失条目；四模型各`0000`–`0239`，共960个唯一的模型×任务记录，同一组240个场景。
- 四个身份fingerprint、checkpoint位置、Base、tokenizer、版本和协议与事前冻结附录一致。SFT30 `best/checkpoint-30`；三个GRPO分别为batch60的checkpoint28/31/31。没有只评validation最高的seed12，也未换成test最高seed11来重定义主比较。
- 随包配置、source lock和bundle manifest与本地原交付一致，原包manifest的全部57文件hash通过；本地55份锁定文件也通过核对。GPU回执均为同一A100-SXM4-40GB、同Base BF16/eager、默认adapter、greedy，无优化器/训练、无自动重试。
- **960/960条原始响应通过现有执行器/checker精确重放**，全部`finished`，infra/缺题/截断均0；四份summary和comparison逐字段复算一致。请求/prompt由冻结执行器从公开任务和工具schema重建，没有把gold/隐藏字段作为模型输入。
- 使用同字节本地tokenizer核对 **2686次生成请求**：生成token IDs→raw text、结束标记、输出tokens和按冻结源码重建的prompt长度均一致。输入/输出tokens不是估算值。
- 远端`readback.json`记本地和Drive文件集合及hash相同，四组240、infra0、complete=true。**本地独立复核了ZIP内960条记录，但未另行连接Drive取第二份副本**；Drive相等仍属于已核源码所产生的远端读回回执。
- manifest为completed、test_consumed=true，一次session，无resume。结束GPU快照0MiB、无进程。日志有`torch_dtype`弃用提示，无Traceback/OOM；没有单独归档进程exit文件，不将它表述为额外核验了exit=0。

冻结配置中的`gpu_execution_authorized=false`和binding中的`preflight.GPU_started=false`是**准备阶段历史记录**，不是本次实际执行状态；GPU由另行确认的命令行口令/预算授权，实际loader回执`real_model_loaded=true`，最终`test_consumed=true`。旧开发入口及其拒绝test规则未被放开。

机器证据位于`results/research/final_test_feedback_review_v1/`：[`verification.json`](../results/research/final_test_feedback_review_v1/verification.json)、`metrics.json`、`paired.json`、`strata.json`、`costs_and_robustness.json`、`cases.json`、`per_scenario.csv`。这些是最终派生证据；原始反馈ZIP未随Git发布，完整复核的输入边界见[复现说明](REPRODUCIBILITY.md)。

## 2. 主表与同题配对

答案、strict、semantic分母均240；动态依赖实际分母为 **144**，不是validation的106。strict是参考轨迹完整性，不等于答案正确；合法一步替代路线可获答案/semantic分，不被包装为参考动态链有效。

| 模型 | 答案正确 | 正确率 | strict | semantic | 动态依赖 | 平均工具调用 |
|---|---:|---:|---:|---:|---:|---:|
| SFT30 | 144/240 | 60.00% | 147/240 | 151/240 | 60/144 | 1.692 |
| GRPO seed11 | 181/240 | 75.42% | 189/240 | 193/240 | 102/144 | 1.871 |
| GRPO seed12 | 171/240 | 71.25% | 176/240 | 179/240 | 90/144 | 1.821 |
| GRPO seed13 | 170/240 | 70.83% | 175/240 | 180/240 | 88/144 | 1.808 |

| 相对同一SFT30 | 变对 | 变错 | 净增题数 | 增量（百分点） | 配对95%区间（百分点） |
|---|---:|---:|---:|---:|---:|
| seed11 | 38 | 1 | +37 | +15.42 | [10.42, 20.00] |
| seed12 | 29 | 2 | +27 | +11.25 | [6.67, 15.83] |
| seed13 | 27 | 1 | +26 | +10.83 | [6.67, 15.00] |
| 三seed平均 | — | — | 平均+30 | **+12.50** | **[8.33, 16.67]** |

主区间按每题`(seed11+seed12+seed13)/3 − SFT30`形成240个值，沿用事前固定seed20261011、1000次percentile场景配对bootstrap；四个模型在同题上共同重采样。各seed差值另按同题配对核算，所有改善/回退完整ID保存在`paired.json`。

**这是条件于四个固定模型的任务不确定性区间**，不涵盖SFT训练种子方差或所有未来RL训练方差；每seed/分层是探索性分析，不宣称多重比较校正后的显著性。没有将三模型在同240题上的结果当成720个独立场景。

## 3. 改善在哪里，哪里没有改善？

### 深度与泛化划分（格内为答案正确数）

| 分层 | 题数 | SFT30 | seed11 | seed12 | seed13 |
|---|---:|---:|---:|---:|---:|
| 一步 | 72 | 64 | 64 | 64 | 64 |
| 两步 | 96 | 56 | 74 | 68 | 68 |
| 三步 | 72 | 24 | 43 | 39 | 38 |
| IID | 60 | 56 | 56 | 57 | 57 |
| 未见参数 | 60 | 52 | 51 | 51 | 51 |
| 未见表达 | 60 | 32 | 46 | 44 | 45 |
| 未见组合 | 60 | 4 | 28 | 19 | 17 |

- 三步平均 **33.33%→55.56%（+22.22pp）**；两步58.33%→72.92%；一步不变。
- 组合平均 **6.67%→35.56%（+28.89pp）**，明显优于SFT，但三个seed都不到50%，不能写“组合能力已解决”。
- 未见表达53.33%→75.00%；参数86.67%→85.00%，三个seed均有同一题回退。
- 该test的组合题只有`measurement_scale`24题和`distance_offset_convert`36题，不代表任意未见工具组合。划分间家族构成不同，不能把分层差异解释为纯单因素效应。

### 全家族表（不只展示受益家族）

| family | 题数 | SFT30 | seed11 | seed12 | seed13 |
|---|---:|---:|---:|---:|---:|
| age_calculation | 12 | 8 | 8 | 8 | 8 |
| country_compare | 12 | 5 | 5 | 5 | 5 |
| country_density_convert | 36 | 24 | 36 | 36 | 36 |
| distance_offset_convert | 36 | 0 | 7 | 3 | 2 |
| measurement_scale | 24 | 4 | 21 | 16 | 15 |
| population_math | 12 | 12 | 12 | 12 | 12 |
| search_and_convert | 12 | 8 | 10 | 8 | 8 |
| simple_convert | 15 | 12 | 12 | 12 | 12 |
| simple_lookup | 15 | 10 | 10 | 10 | 10 |
| simple_math | 15 | 15 | 15 | 15 | 15 |
| simple_search | 15 | 15 | 15 | 15 | 15 |
| simple_weather | 12 | 12 | 12 | 12 | 12 |
| weather_compare | 12 | 7 | 6 | 8 | 8 |
| weather_convert | 12 | 12 | 12 | 11 | 12 |

平均净增30题：density贡献12题（40%）、measurement_scale贡献13.33题（44.44%）、distance_offset贡献4题（13.33%），三者合计 **97.78%**。三seed共同变对24题，分别为density12、measurement10、distance_offset2；并非全来自density。

作为**事后探索性敏感性分析**，排除density的204题中，SFT为120，三seed为145/135/134；均值仍提升8.82pp，配对场景区间[5.07,12.75]pp。这不能替代预先规定的完整240题主比较，也不是新的选题规则。

### 错误类别

| 多标签错误次数 | SFT30 | seed11 | seed12 | seed13 |
|---|---:|---:|---:|---:|
| unit_error | 19 | 7 | 13 | 6 |
| value_error | 76 | 52 | 56 | 64 |
| missing_step | 76 | 33 | 47 | 48 |
| wrong_arguments_or_route | 13 | 14 | 14 | 12 |
| tool_execution_error | 4 | 2 | 3 | 3 |
| parse_error | 1 | 0 | 0 | 0 |

错误是多标签，不可相加当作错误任务总数。单位/缺步减少，但参数路线错误没有一致改善。

## 4. 成本与预算

| 模型 | 工具调用总数 | 生成请求 | 生成tokens | 输入tokens | 任务推理耗时（秒） |
|---|---:|---:|---:|---:|---:|
| SFT30 | 406 | 646 | 20010 | 498788 | 1205.15 |
| seed11 | 449 | 689 | 21771 | 536809 | 1316.57 |
| seed12 | 437 | 677 | 21399 | 526392 | 1293.70 |
| seed13 | 434 | 674 | 21231 | 523792 | 1284.29 |

GRPO相对SFT的三seed平均：工具调用 **+8.37%**、生成tokens **+7.28%**、输入tokens **+6.06%**、任务推理耗时 **+7.72%**。平均调用1.692→1.833，不能声称更省调用或推理更快。

四模型共同正确的同142题，调用237→241/240/240、生成tokens12105→12205/12180/12180，仍略增。该事后共同成功子集只用于配对成本描述，不用各自不同的成功题均值冒充效率增益。

整个评测进程 **5185.73秒=1小时26分25.73秒**，在用户确认的10800秒上限内；逐题推理合计5099.71秒，其余约86.02秒含预检/加载/同步/读回等。没有逐模型加载/同步分项，不能拆造精准阶段时间；安装/下载/idle不在进程wall里，**不能据此声称Colab当前还剩精确多少小时**。

逐题CUDA峰值allocated约1.093GiB，reserved约1.600GiB；实际最大prompt951 tokens、单turn生成58 tokens、总请求输入+输出最大976 tokens、最长4turn。没有OOM，但这不是4096-token长轨迹压力测试，也不是整块GPU所有进程/全会话的采样峰值。结束快照0MiB/无进程不等于已删除Colab运行时；如仍开启请释放。

## 5. 可核查案例（成功与回退都保留）

完整消息和trace见`cases.json`，以下短ID均能对应完整ID：

- `rv1-7800ed53…`，density/未见表达：SFT查询人口和面积并计算出8230.4222人/km²后提前结束；三个GRPO都补上km²→mile²转换，输出21316.695699人/mile²，答案与三步参考链均正确。
- `rv1-90b89998…`，measurement/组合：任务要求距离108.7km乘4。SFT只search后输出434.96；三个GRPO调用calculator `108.7*4`，得到正确434.8km。
- `rv1-7a9dabe0…`，distance_offset/组合：SFT完成168.5+7=175.5，却把175.5直接标成miles；三个GRPO继续convert_units，正确输出109.050644 miles。
- **共同回退** `rv1-f6177dd4…`，weather_compare/未见参数：四个模型都查到了两个温度、参考链均完成，SFT选对实体，三个GRPO都选错。说明流程正确不能替代答案正确。
- **seed12额外回退** `rv1-b78d912d…`，weather_convert/未见表达：SFT/seed11/seed13直接合法查询Fahrenheit，seed12却返回Celsius，遗漏转换。全test回退题union仅这两题，不能隐去。

## 6. 与validation的关系与最终口径

| 模型 | validation正确率 | test正确率 |
|---|---:|---:|
| SFT30 | 71.11% | 60.00% |
| seed11 | 75.56% | 75.42% |
| seed12 | 78.33% | 71.25% |
| seed13 | 75.56% | 70.83% |
| 三seed均值 | 76.48% | 72.50% |

主改善由validation的+5.37pp变为test的+12.50pp，但两个split的组合比例和家族/实例构成不同，不把差值增大解释成训练越多越好或test更简单。validation最高seed12并非test最高seed；预先保留三seed平均避免了事后挑最高模型。

**可用于最终报告的表述：**

> 在冻结模型和评测协议后，答案主导GRPO在240个独立合成test场景上，三个训练种子的答案正确率均优于同一SFT30基线；平均从60.00%提升到72.50%（+12.50个百分点，任务配对bootstrap 95%区间8.33–16.67个百分点）。改善主要体现在若干两/三步任务家族及所测组合/表达泛化，伴随约8.37%的工具调用增量，未表现为全面能力或效率提升。

继续保留：seed11训练轨迹476/480、总1436/1440的四条日志缺口；采样与teacher logprob差异；只有一个SFT训练seed和三个RL seed；合成基准/小分层及非外部盲测；远端权重未下载到本地独立加载。test好成绩不消除这些局限。

本次最小四模型研究闭环可以收尾，无需追加SFT/GRPO或补训缺失日志。后续若研究新方法，应另立实验与新的独立验证安排，不再消费这个test来追分。

## 7. 原始结果复核

入口：`scripts/review_research_final_test_feedback_v1.py`，使用既有HF CPU环境、`CUDA_VISIBLE_DEVICES=''`、`HF_HUB_OFFLINE=1`。显式指定原ZIP、固定本地Base缓存和一个**不存在的新输出目录**即可复算；不会调用GPU或模型forward，也不会改原ZIP/冻结配置。

审查器从**冻结执行器重建工具schema的原插入顺序**，而不是直接重渲染经过sort_keys的保存JSON；后者会改变prompt。核对请求语义和原始响应后，2686次prompt长度与远端usage一致。prompt输入IDs未单独保存，长度一致与源码重建不等于独立采集了远端输入IDs。逐题CSV统计复算无需这些原始产物，具体命令见[复现说明](REPRODUCIBILITY.md)。

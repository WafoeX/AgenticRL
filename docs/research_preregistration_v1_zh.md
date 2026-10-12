# Research V1 决策预注册与操作性 test 冻结

注册 ID：`research-preregistration-v1-20261012`。用户于本轮明确确认五臂、单种子、消融、选点及失败处理，并选择独立原字节冻结副本。机器决策为 `configs/research/preregistration_v1.json`；登记锚点为同目录 `preregistration_v1.lock.json`；冻结发布记录为 `data/research_v1_frozen/freeze.json`。

**这是模型正式对照前的本地决策预注册，不是外部时间戳注册、盲测认证或训练授权。R0.3 的决策协议与 R1.8 数据冻结在本轮锁定；数值训练预算/配置附录尚待成本 pilot，附录齐全并另行授权前不得训练或最终 test 推理。** 用户拥有本地文件，可以协同改写所有锚点，SHA-256 只能检查相对于保留锚点的身份，不能提供不可抵赖签名或操作系统访问隔离。未自动 commit/push。

## 1. 主问题、已知观察与假设

研究问题：小参数工具模型的 SFT warm-start 与研究版轨迹奖励，是否改善独立任务最终答案及多步执行，而不仅提高 reward。

本次注册在已完成数据构建、程序化审计和两个 validation 场景的 Base CPU pilot **之后**；该 pilot 的 greedy0/2、sample0/16 已知。不是声称未见任何开发结果的纯事前注册。没有已知模型 test 推理记录。研究允许无提升或负结果，不承诺显著性或最小涨幅。

主要假设为 G2 相较 G1 的最终 test greedy 答案正确率改善；报告双向配对差值/区间，不预设结果。A1 相较 G2 用于检验动态依赖奖励的作用线索，不把消融结果解读为纯因果机制。B0→B1 描述 warm-start 的探索性差异；合成环境、单种子和有限实例不支持普遍结论。

## 2. 固定数据与可见性

- 原 candidate 六文件保留原样，不改变记录内 `dataset_version=research-v1-candidate` 或原 manifest 的 `candidate_not_frozen` 历史字段。
- 独立复制为 `data/research_v1_frozen/`，额外 `freeze.json` 是冻结发布的权威登记；release ID `research-v1-frozen-20261012` 不伪装成重新生成数据。
- train1200 / validation180 / test240；各 split 参考深度30%/40%/30%，test四泛化层各60。原分层、recipe/wording/参数注册与审计不变。
- test 原始 SHA-256：`08040a4e2e7b7c410f17937496a22bccd09f925e4c5dba35da2f027380410a52`。其他文件 hash、字节数和实现工作树 hash 见锁与原 manifest。
- test 已经过程序化构建、格式/可解性/容量/分组泄漏审计；标签、隐藏事实、参考轨迹在本地文件中可访问。只称**自建操作性 holdout**，不称外部盲测、所有事实未见或纯单因素参数 OOD。
- 本轮冻结只逐字节哈希/复制/计行，不解析 test task/gold/trace，不做模型推理。现有开发评测仍只接受 train/validation；新 `load_registered_development` 先检查锁和源身份，再只提供 train/validation。
- 训练/专家数据只能来自 train；validation 可选点与开发，不能转为监督训练。共享工具环境不等于训练标签，但潜在事实记忆捷径继续披露。

## 3. 五个实验臂与公平控制

| 臂 | 固定含义 |
|---|---|
| B0 | 固定官方 Qwen/Qwen2.5-0.5B-Instruct，无项目 LoRA |
| B1 | Base → train-only SFT，validation 选点 |
| G1 | 所选 B1 → 独立 GRPO V1 |
| G2 | 同一 B1 → 独立 GRPO 研究版 V2 |
| A1 | 同一 B1 → 独立 GRPO 研究版 V2，移除动态依赖奖励 |

Base revision `7ae557604adf67be50417f59c2c2f167def9a775`；单种子20261011。不自动追加种子。G1/G2/A1 初始 checkpoint 哈希必须相同，优化器独立初始化，禁止接着另一臂续训。匹配数据/世界/prompt/template、场景顺序/seed、计划更新与组大小、采样/token/turn限制、评测频次/选点和 infra 规则。B1 的额外 SFT 成本另报，不说五臂总训练成本相同。

A1 只移除动态依赖分项，不补偿或重新缩放剩余奖励；其余分项和超参数与 G2 相同。奖励总量及方差可能同时变化，因此不是纯粹机制识别。研究版 V2 需要适配新 recipe，不能称与历史 V1 只相差一个分项；A1 的单项差异只针对同一冻结研究 V2。

## 4. 指标、选点与统计

主终点：最终 test greedy 答案正确场景数 / 全部240指定场景。oracle、语义路线和严格协议分离，合法替代答案获答案信用，不以 reward>阈值判成功。

每个候选 checkpoint 在**完整180条 validation greedy**选点，按以下固定字典序：

1. 答案正确场景数降序；
2. reference dynamic dependency 有效场景数降序（固定需要动态依赖的子集，合法绕过单列）；
3. checkpoint step 升序。

同一 run 每 step 只接受一个明确身份；不根据 test、sample@8 或 train reward 选点。保存 best/last。若一个 validation run 存在 infra/缺失，它不能进入候选 ranking，也不能无条件当最差 checkpoint；先按下节留账和修订规则处理。候选 step 清单与验证频次在训练前预算附录锁定，不能边看曲线边增加选点评测。最终选定的每臂模型才执行完整 sample@8。

评测沿用 `research-eval-v1`：greedy temperature0/top_p1，sample temperature1.2/top_p.95、k8；6 turns/8 calls，512 tokens/turn、2048输出/attempt、4096输入/request、60s软限、并发1、精确模板和 Qwen 文本 codec，逐turn seed由当前固定协议派生。温度0不保证跨环境字节确定性。目标 GPU 后端还未验证，若必须更改并发/解码/协议，先版本化修订并重新验证，不静默替换。

最终五臂 test 单种子固定240×9×5=**10800 planned attempts**。一次性锁定全部臂身份/选点/协议后才能开始其中任一臂 test；不能先看一臂结果再改其他臂。

报告 greedy 答案、答案 Pass@8、sample rollout、strict/semantic/dynamic、合法绕过、family/depth/generalization 分层、失败计数和成本。给分子/分母及场景 Wilson95%区间；sample/模型配对差值按语义场景聚类 bootstrap，seed20261011、1000 replicates；八次采样不是八个独立任务。主比较G2−G1；其他比较/分层探索性报告，不做挑选最好分层的确认性显著性宣称，不将单种子解释为训练稳定性。

## 5. 失败、重试与停机纪律

- 评测：所有计划 attempts 保留；错误参数、错误工具、空模型文本、解析/单位/预算失败是模型错误，不当 infra 过滤。infra/真正缺失使对照无效。
- 不自动重试、不追加成功样本；如需重跑，先写版本化修订并获用户授权，以新 cohort/run 对全部受比较臂完整计划重跑，保留原失败结果与额外成本。
- GRPO：按同 prompt 完整组处理 infra；最多一次整组重试，仍失败跳过整个组，并保存两组全部成员/原因/计数；不删除单成员拼不完整组，不因低奖励/错误工具过滤模型失败。零方差组如实记录，不当作学习信号。
- 上述 GRPO 策略是**预注册要求，尚未实现/验证**。重试 seed/场景顺序、最大跳组比例、计划/有效预算偏差阈值需在 R4 和预算附录提前锁定。发现后端实际行为不符，停在开发 pilot，不进入正式对照。
- 硬中止：即将超已确认 GPU 余额、身份/hash/协议不符、test误用于调参、无可训练参数/无监督 token、非有限 loss/grad、保存/恢复污染，均停止并留账。非零warning与infra不静默忽略。具体性能/跳组/耗时数值阈值待测成本后锁定，不臆造。

## 6. 预算附录与禁止越关

资源约束：Colab A10040GB、总20GPU小时、2026-10-25目标；准备/分配/空闲/pilot/失败重试/训练/评测都计入。历史已用时间及当前余额未重新核实，本次注册GPU0，不把余额自动写成20小时。

正式训练前必须版本化锁定：

- 实际 GPU 分配、历史用量/余额、Drive备份及中断恢复；SFT/PEFT/ART-vLLM兼容与成本 pilot。
- SFT学习率/epoch/step/batch/LoRA/assistant-only mask/截断；GRPO有效学习率/组大小/rollout/更新/KL等；V1/研究V2/A1实现hash、分项及范围。
- 同起点hash、候选checkpoint/validation频次、每臂计划预算、重试seed/上限、跳组和有效预算偏差容忍度/停机阈值。
- 包括10800次test在内的总成本可行性；如果不够，**在任何模型test前**请用户缩范围并发布修订，不静默缩分母。
- 最终测试就绪附录：全部臂已选身份、数据/环境/schema/checker/reward/协议hash、完整cohort计划、退出/清理留档和单独授权。

当前 `formal_training_ready=false`、`final_test_ready=false`。本注册不启动 GPU、训练、SFT专家构建或模型推理，也不实现最终 test 解锁入口。R0.3/ R1.8 完成不等于 R3/R4 或完整实验就绪。

## 7. 复核、变更与误消费

```bash
# 仅第一次、用户已授权的数据复制/登记；已有release或lock时拒绝
python3 scripts/freeze_research.py --create
# 后续只读校验，不能覆盖或重生成冻结记录
python3 scripts/freeze_research.py --check
```

冻结锚点绑定原六文件、额外freeze.json、决策JSON、本文及生成/环境/评测/冻结实现的工作树hash。已有结果和原candidate保留；未来新增R3源码可以新增独立文件，但被锁源码更改需版本化修订，不覆盖本锁来把旧身份冒称新身份。

未来消费者应使用 `load_registered_development` 或先执行 `--check`；当前旧入口的 train/val API 本就拒绝 test，但并未声称所有任意用户脚本都被系统拦截。冻结目录与test标签没有加密/权限隔离，用户可直接打开；主要保障来自哈希审计和操作纪律。

修订必须追加新文件/版本，说明日期、原因、受影响协议/预算/源码/数据hash以及是否已看test；保留本注册原锚点和此前失败记录。若看模型test成绩后改模型/奖励/选点/协议，本版本test已消费；不能继续主张未见最终测试，需新的研究版本/holdout，并披露旧消费历史。

# AgenticRL 算法实习简历：研究闭环任务书

> 建立日期：2026-10-11。目标：把已跑通的工具调用训练工程原型，完善成可复现、可解释的小型算法研究项目。
> 本文件是执行与验收清单，不是实验结果报告。未勾选项均未完成；拟新增路径不代表已有实现。
> 当前已完成旧模板审计、候选数据容量/泄漏、R1.6/R2.1–R2.4 checker，以及 R2.5–R2.8 本地评测入口的 CPU 合同验收。2026-10-12已锁定R0.3决策预注册与R1.8操作性test冻结，预算/训练配置附录及最终test就绪仍待锁定；Base CPU validation pilot v2 已完成真实身份加载/生成与 raw 重放，但两场景未答对、正确多步/真实checkpoint/ART-vLLM兼容仍待验。后续记录见第 8 节；本轮无GPU/训练、未覆盖旧数据，未自动提交或推送。

## 1. 目标与完成定义

**研究主问题：在小参数工具调用模型上，SFT warm-start 与轨迹级奖励能否改善独立任务的最终答案和多步执行，而不仅是提高 reward？**

最小完整闭环：发现泄漏/缺步假阳性 → 修正数据和评测 → 建立 SFT 起点 → 等预算 V1/V2 GRPO → 独立测试与失败分析 → 形成可审计结论。

必须交付：

1. 版本固定的 train / validation / test、数据卡与泄漏审计。
2. 独立答案 checker、严格工具协议与语义正确性分离的评测协议。
3. Base、SFT、SFT+GRPO-V1、SFT+GRPO-V2 四组主实验及一项明确假设的消融。
4. 配置、模型/checkpoint 身份、原始结果、训练曲线、失败案例及简短实验报告。
5. 只使用已验证数字的简历表述，以及能解释 GRPO 与实验取舍的面试材料。

**完成不以“必须提升 X%”为条件。** 无提升也应如实分析；只有独立数据上有可比证据，才能写算法效果提升。单种子结果允许作为资源受限的探索，不包装为稳定的普遍结论。

不优先做：新增 Web 界面、接更多外部 API、直接换 7B、无对照增加训练步数、从零重写 GRPO。现有生命周期告警可单独维护，但不以重复 smoke 替代算法实验。

## 2. 起点：已有什么、还缺什么

依据本地代码 `09e2905f08e3e3d5a9dde8940ad782a4c08e4c35`、本次数据复算，以及 [生命周期反馈复核](lifecycle_feedback_review_20261011_zh.md)：

| 项目 | 已知状态 | 不能推导出的结论 |
|---|---|---|
| 模型与训练链路 | Qwen2.5-0.5B-Instruct、ART 0.5.17、LoRA、GRPO；真实 2-step smoke 有更新与 checkpoint 加载证据 | 没有完成有意义的学习曲线或训练收益对照 |
| 工具与任务 | 5 类本地确定性工具；旧数据为 12 类模板 | 不是生产外部 API Agent，也不是独立 benchmark |
| 数据 | train 200 条 / 86 种文本；val 50 条 / 37 种文本；val 中 36 条、24 种文本在 train 中出现；无独立 test | 旧成绩不能作为可靠泛化证据 |
| 深度分布 | train：1/2/3-call = 82/98/20；val = 23/22/5 | 不同调用次数不等于动态依赖难度 |
| 评估 | deterministic、sample、观测 Pass@8、trace、V1/V2 同轨迹评分、严格完整链路检查已具备 | 缺独立最终答案判定；同轨迹重评分不是训练对照 |
| 最新 smoke 能力 | 50 场景中严格链路深度 1 成功 13/23，深度 2 为 0/22，深度 3 为 0/5；sample 未建立多步完整成功 | 不能写“显著提升多步推理能力” |
| 工程验收 | 此次 Colab A100 上 173 tests pass；训练、单场景评估、450-rollout 审计均 exit=0，结束后 GPU 查询为空 | 不是所有环境/取消路径全覆盖；仍有 NCCL/semaphore 告警 |
| SFT/受控研究 | 尚无 SFT 入口、共享 SFT 起点的 V1/V2 训练对照与消融证据 | 不把计划写成成果 |

旧数据哈希（仅用于识别旧版本，不作为新实验数据）：

- `data/scenarios_train.jsonl`：`c77a225e9e769b887422d74ac56f1d82011592299404ac8c08d4dec1c29c53c7`
- `data/scenarios_val.jsonl`：`cdf7ef85605da97f131994993f1c6a91ef290931a37bf8b450e1f7a12fd2fe6c`

旧 README 与 [既有简历介绍](resume_project_intro_zh.md)含历史状态和旧 checkpoint 数字，不作为新研究完成证明。不同 smoke checkpoint 的百分比不直接比较，不自动宣布整个 Phase 3/4 验收。

## 3. 总路线与关卡

这里使用 R 编号，避免与仓库旧 Phase 1–5 混淆。

| 阶段 | 内容 | 优先级 | 通过后允许做什么 | 当前状态 |
|---|---|---|---|---|
| R0 | 确认研究范围、GPU预算与预注册草案 | P0 | 确定实现规格与试跑上限 | 范围/资源/决策预注册已锁定；正式训练预算附录待成本pilot |
| R1 | 数据语义修复、分组去重与三份数据 | P0 | 开发正式独立评测 | R1.1–R1.8通过；独立原字节操作性冻结，模型test仍关闭 |
| R2 | 答案 oracle、指标协议与固定评测入口 | P0 | 单独授权的小规模开发加载/推理 pilot，不使用 test 调试 | R2.1–R2.8本地/CPU合同通过；Base CPU pilot通过加载/生成/重放，正确多步/checkpoint/目标后端待验 |
| R3 | 专家轨迹、SFT 与 ART 起点兼容验证 | P0 | 正式 SFT / GRPO 实验 | R3.1–R3.3本地CPU/tokenizer通过；训练/兼容待验 |
| R4 | GRPO 配置生效、基础设施失败隔离 | P0 | 等预算 V1/V2 对照 | 未开始 |
| R5 | 四组主实验与 validation 选点 | P1 | 一项预注册消融与最终 test | 未开始 |
| R6 | 消融、可选多种子与冻结测试 | P1 | 汇总可信结论 | 未开始 |
| R7 | 结果表、曲线、案例、报告与简历 | P1 | 对外展示研究贡献 | 未开始 |

依赖：R0 → R1 → R2；R1/R2 → R3；R2 → R4；R3/R4 → R5 → R6 → R7。每轮只推进一个可验收任务，失败先定位，不越过关卡。

## 4. 分阶段任务与验收

### R0：先确认范围与资源

- [x] R0.1 确认推进完整研究闭环，沿用当前小模型与四组主对照，不优先扩展产品功能。证据：2026-10-11 用户问卷选择“完整研究闭环（推荐）”。
- [x] R0.2 确认可用 GPU/显存、总 GPU 小时上限与期限。证据：用户明确“Colab A100 40GB，总计20小时，两周内”。
- [x] R0.3 写并锁定决策预注册：主指标、split、B0/B1/G1/G2/A1、validation字典序选点、移除动态奖励消融、单种子20261011及评测/整组infra规则。证据：[正式决策协议](research_preregistration_v1_zh.md)、`configs/research/preregistration_v1.json`及独立hash锁。用户问卷确认；训练步数/频次/有效配置/奖励hash和具体GPU成本分配按原任务允许留为开发pilot后的训练前附录。formal_training_ready=false、final_test_ready=false，不是训练授权或外部盲测注册。
- [x] R0.4 确认数据规模与各分层最低样本量。用户已接受目标 train1200/val180/test240、test四类各60、整体深度30%/40%/30%；候选数据已实际构建且全部配额一致，40个需求池各1024独立值/操作组，最紧池需求411、余613。证据：[CPU实现验收](research_dataset_implementation_zh.md)。family细分配额公开，但小分层样本少，不代表统计功效充分或test已冻结。

**交付：**研究协议草案 + 资源约束表。**验收：**主问题明确，数据/实验规模可行；正式训练前须补齐并锁定预算和选择规则，不猜用户预算。

已确认的资源约束：

| 项目 | 约束/执行原则 |
|---|---|
| 范围 | 完整研究闭环：四组主对照 + 一项消融 + 报告 |
| GPU | Colab A100 40GB；运行时实际分配需检查 |
| 总预算 | 20 GPU 小时上限；pilot、SFT、GRPO、评测、失败重试均计入，GPU 分配后的准备/空闲时间也留账，不只统计优化器时间 |
| 时间 | 两周内；按本次确认日暂以 2026-10-25 为目标，不是自动定时执行安排 |
| 种子 | 优先完成一套可信单种子闭环；2–3 种子仅在成本实测与剩余预算允许时增加 |
| 待确认 | Colab 中断/恢复条件、Drive 可用容量与备份路径，在 GPU pilot 前检查 |

建议节奏（目标安排，不是耗时保证）：第 1 周完成数据/评测、专家轨迹与加载 pilot；第 2 周进行锁定预算的四组对照、消融、冻结测试和报告。先测算单次成本再分配 20 小时；预算内不能保证正向增益或多种子。若兼容性/容量超出预算，停下说明 blocker 并由用户决定缩减范围，不自行超时。

### R1：可信数据，而不是简单换 seed

现有入口：`src/scenarios.py`、`src/tools.py`、`scripts/generate_scenarios.py`、`data/`。新数据应写入独立版本目录，保留旧数据与反馈可复核。

- [x] R1.1 审计全部模板：任务可解、知识库覆盖、量纲、预期调用与答案捷径；记录新研究版修复/替换决定。证据：[12类模板审计报告](research_dataset_audit_zh.md)、机器JSON、21项新增测试；Guido检索错配、密度量纲错误、两类比较固定slot B已复现。实际修复随新版本R1.2–R1.7实施，旧模板未修改。
- [x] R1.2 定义结构化记录：唯一 sample ID、任务家族、语义实例/参数、表达变体 ID、工具深度、依赖图、split/泛化类别、环境版本、答案类型/单位/容差、oracle 与参考轨迹信息。证据：独立JSON Schema、运行时校验，1620记录正式schema验证通过；oracle字段是内部公式gold，不代表R2 checker完成。
- [x] R1.3 先分配语义实例组再渲染表达。同一参数实例的改写、归一化重复、等价预期调用不能随机散落 train/val/test。既做文本审计，也做语义/参数/轨迹签名审计。证据：精确/规范化文本、语义组、去实体名值、完整轨迹跨split及内部重复均0；审计重推导记录，非仅信任保存hash。
- [x] R1.4 划分并分别报告：同分布未见实例、未见参数/实体、未见表达模式、未见组合。新组合应由训练中已学的原子工具组成；不要把“模板完全未见”和“参数未见”混称。证据：val90/30/30/30、test四类各60，参数/wording/recipe注册隔离；原子工具/运算/转换训练覆盖。实体新颖性与共享train observation事实另列，不宣称纯单因素OOD或所有事实未见。
- [x] R1.5 覆盖 1/2/3-call 与真正动态依赖，并标记“两个独立查询”和“后一步消费前一步 observation”。对有限枚举模板检查容量，不足时扩展可解的实例空间或减小规模。证据：三split均30%/40%/30%，参考真实执行并消费已返回observation，双比较无动态边；容量完整枚举并足额构建。weather_convert可一调用，基础三步仅密度，局限已披露。
- [x] R1.6 动态依赖开发集加入受控 observation 扰动/环境变体，测试答案与后续参数是否跟着变化。新实例只能暴露工具信息，不能把预期调用、gold answer 或隐藏元数据拼进模型 prompt。证据：[配对干预/checker验收](research_checker_interventions_zh.md)，validation106组scripted正控全部消费变化，各4类负控106/106拒绝；10组Fahrenheit合法绕过单列。prompt不变、不可变世界隔离、test输入拒绝；不是模型因果/能力成绩。
- [x] R1.7 输出数据卡、split manifest、生成 seed/代码版本、SHA-256、每类/每深度/每泛化分层计数；校验重生成一致。证据：`data/research_v1_candidate/`六个候选产物，`--check`重生成全部字节一致；代码commit仅基准，未提交源码另有工作树hash。原candidate及其历史manifest不改；独立冻结发布见R1.8。
- [x] R1.8 冻结 test及访问纪律；`data/research_v1_frozen/`原六文件逐字节等同candidate，独立freeze.json登记train1200/validation180/test240与历史可见性，锁绑定决策/数据及20份实现/协议文档hash。test原hash保持；仅操作性holdout，标签本地可见、无模型test推理。`freeze_research.py --check`只读、拒绝覆盖/篡改；开发加载/计划仍拒绝test。证据：[冻结协议](research_preregistration_v1_zh.md)、[验收记录](../results/research/registration_v1_20261012/REPORT.md)，26新增测试/506passed；正式test需另行就绪附录和授权。

**交付：**版本化 train/val/test、数据卡、泄漏审计脚本和测试。冻结目录已定为 `data/research_v1_frozen/`，原 `data/research_v1_candidate/` 保留；冻结登记在额外freeze.json，不重写历史数据卡/manifest。

**验收：**精确/归一化文本跨 split 重叠为 0；按预注册分组规则禁止的语义实例跨 split 重叠为 0；所有 oracle 与参考轨迹可执行；无不可解任务、错误量纲或静默丢样本。各泛化分层单列，不能靠一个总百分比掩盖重复或容量不足。

**测试纪律：**先在开发数据验证代码，再冻结测试。自建且能查看的 test 只是操作性 holdout，不宣称外部盲测。若看 test 后修改模型/奖励/协议，该版本 test 已被消费；后续探索用 val，需新版本才能再主张未见测试。

### R2：独立答案校验与评测协议

现有 `scripts/eval_checkpoint.py` 和 `src/diagnostics.py` 可扩展，但现有完整链路 checker 是严格协议检查，不是最终答案 oracle。

- [x] R2.1 对各任务定义最终答案结构与解析：数值、单位、枚举/比较结果、天气字段等；提前固定容差、同义形式、多答案冲突和解析失败规则。证据：`research-answer-v1`严格value/unit JSON、终止assistant适配、重复键/多答案/类型拒绝；整数精确、其余abs1e-6/rel1e-9，单位/实体无隐式别名；当前天气任务为温度，未新增condition任务。
- [x] R2.2 建立不依赖 V1/V2 reward、也不依赖模型自身回答的程序化 oracle。转换公式/平方单位等用独立测试验证；若共享工具实现，披露共同错误风险并用手算样例交叉检查。证据：纯stdlib `src/research_answers.py`重算16recipe原始事实；不导入工具/生成器/reward、不信任保存gold/trace/容差；手算SI与平方密度、毒化gold和破坏executor测试通过。共同事实/任务定义风险已披露。
- [x] R2.3 主指标为最终答案正确率；另外报告严格工具链和语义/依赖有效性。直接得到正确答案的合法替代解法应计答案正确，不因不符合参考顺序而判答案错。证据：`research-trace-v1`输出答案、真实执行、严格链路、语义路线及动态状态；无工具正确答案也获答案信用，Fahrenheit替代不获参考动态边信用。模型主指标尚未运行。
- [x] R2.4 测试 adversarial trace：缺步高分、错误参数、虚构 observation、同 turn 伪依赖、单位错误、答案正确但协议不同、协议完成但答案错误、异常与截断。oracle 不允许照抄 reward 规则或仅信任模型输出的 success 字段。证据：[CPU验收](research_checker_interventions_zh.md)，95新增tests、全量370passed；validation180参考通过、106配对及4类负控均通过，raw attempts/manifest可重生成。不是模型端到端或正式R2全阶段验收。
- [x] R2.5 支持显式 base 模型评测及指定 checkpoint；结果记录实际加载身份/step/文件哈希，而非只相信配置名。证据：[本地评测CPU合同](research_evaluation_zh.md)：显式路径/revision、step/Base字节绑定、回执与运行前后哈希核对；模拟Base/step7身份和stub库加载参数通过。2026-10-12真实Base CPU加载pilot回执与运行前后字节检查通过；真实checkpoint加载仍待验，不以假权重代替。
- [x] R2.6 固定 prompt/tool schema、chat template、解码预算、max turns 和并发；记录 deterministic 参数及 sample 参数、k=8。证据：`evaluation_v1.json`、实际请求/CPU生成参数测试、协议hash与逐turn seed；原始EOS/token IDs保留，CPU并发1。温度0不宣称跨环境字节确定性；ART/vLLM适配未验收。
- [x] R2.7 报告全部尝试分母、有效尝试分母、infra failure 和缺失输出；正式比较不静默删失败或追加采样。证据：先保存完整计划、实时raw记录、全量ledger补未返回；加载失败/取消/关闭失败均留账。每模拟身份72计划/72返回/70有效/2infra，comparison无效、零补采样；实际返回空模型文本仍计模型错误，不冒充丢失请求。正式整次重跑须先预注册。
- [x] R2.8 保留每场景 attempt、最终回答、调用/observation trace、各 checker 结果、错误类别、评分器版本。证据：四类分层raw counts、答案/strict/semantic/dynamic分开、场景Wilson/cluster bootstrap/配对差值；生产离线raw重放在临时fixture通过，脱敏导出由顶层manifest/--check复核。100新增tests、全量470passed；不是模型能力或训练增益。

**交付：**评测协议、独立 checker 与单元/集成测试、开发集端到端报告。**验收：**人工可核对案例和对抗样例全部通过；修改 reward 不改变最终答案判定；Base/各 checkpoint 同一入口评测且身份可追溯；无分母缩水。

指标与分母必须固定：

| 指标 | 口径 |
|---|---|
| Deterministic 答案准确率（主指标） | 一次解码答案正确场景 / 全部指定场景 |
| 观测答案 Pass@8 | 8 次采样至少一次答案正确场景 / 全部指定场景；不是 8 条平均成功率 |
| Sample rollout 答案正确率 | 正确 attempt / 全部指定 sample attempt |
| 严格完整链路与其 Pass@8 | 使用已标记版本的协议 checker，独立于答案与奖励 |
| 动态依赖有效率 | 在需要动态依赖的场景上定义；分母与时序/扰动检查标准写清 |
| 失败与诊断 | 参数错误、缺步、工具执行错误、无工具、答案解析失败、超预算、infra；可多标签，互斥主错误另定优先级 |
| Reward hacking 审计 | reward 高但答案错、reward 高但链路未完、答案对但协议不符；不把它们混为一类 |

同时给 raw counts；主要准确率给置信区间。对同一任务的模型差异优先配对分析，Pass@8 按场景聚类，不把 8 条轨迹当 8 个独立任务。分层样本少时披露区间与局限。

### R3：真实专家轨迹与 SFT warm-start

- [x] R3.1 仅从 train 构建监督轨迹：执行工具 → 记录真实 observation → 填入动态参数 → 下一 turn → 最终答案；不得用 `__dynamic__` 或未经执行的静态拼接充当专家轨迹。证据：[专家/SFT数据验收](research_sft_data_zh.md)，冻结train1200条重新执行，独立`data/research_sft_v1/`保存messages/trace，未复制静态gold/参考轨迹。
- [x] R3.2 逐条通过独立 oracle、执行/依赖检查；排除不可解或错误轨迹并记录数量与原因。证据：1200/1200答案、真实执行、strict/semantic通过；动态边valid，独立比较not_applicable；拒绝0，逐ID拒绝台账与gold毒化/伪observation等负控通过。共享executor风险已披露，不是模型成绩。
- [x] R3.3 定义 SFT chat template、tool-call 序列化、assistant-only loss mask；用户消息/工具 observation 不误作为预测监督。验证多轮数据、EOS、截断与 token 长度。证据：固定revision真实Qwen tokenizer/原模板，监督assistant payload+EOS，context/header/padding=-100；739–969tokens、监督120963tokens，超4096整条拒绝、不裁剪。多轮/边界/长度/EOS/padding测试、两次完整字节--check通过；26新增、全量532passed，未训练。
- [ ] R3.4 实现/选择 SFT 路径并做小型兼容 pilot：SFT LoRA/合并产物能否被 ART/vLLM 正确加载，tokenizer/schema 一致，加载前后行为/权重身份可核对。**本地入口部分已实现**：[入口合同验收](research_sft_entry_zh.md)、[Colab独立授权指南](colab_research_sft_pilot_zh.md)；3条train、两次LoRA更新的固定兼容合同，CPU随机微型模型与真实tokenizer测试通过、全量567passed。尚无官方权重GPU更新/Transformers重载或ART/vLLM结果，因此不勾选。
- [ ] R3.5 固定 SFT 配置与 validation 选点规则，保存实际更新/训练 loss；验证不是空 loss、零可训练参数、截断后无动态消费 turn。
- [ ] R3.6 将同一个已选 SFT checkpoint 复制/引用为两条独立 GRPO run 的初始权重，校验哈希；禁止 V2 接着 V1 训练，禁止混用旧 smoke 或同名 `.art`。

**交付：**可复现专家数据、SFT 入口/配置、加载兼容测试、SFT checkpoint manifest。

**验收：**轨迹真实可执行且无 split 泄漏；SFT 实际更新、评测加载成功；两个 GRPO 分支的起始权重与模板完全一致。warm-start 是否有效由实验判断，不先承诺提升。

### R4：GRPO 的可控训练与失败隔离

- [ ] R4.1 逐项验证配置生效：seed、学习率、rollout/group 大小、temperature、token/turn 预算、LoRA 设置、reward_version、KL/beta。当前 YAML 中并非所有字段都已接入，需记录运行时实际值。
- [ ] R4.2 保留 ART 训练需要的原始 Choice/token logprobs，核对多 turn、tool observation 与训练 token mask；不因评测 trace 重构丢失训练信息。
- [ ] R4.3 区分 infra 与真实模型/工具错误；只有基础设施失败隔离，不能将错误参数、错误工具或低分样本过滤掉。
- [ ] R4.4 预注册按同一 prompt 的固定组大小处理策略：建议整组有限重试，重试后仍有 infra 则跳过整组并记录，禁止删个别成员后把不完整组送入 GRPO。检查 ART 实际组处理行为与保留样本偏差。
- [ ] R4.5 记录请求/有效 rollout、重试/跳过、每组 reward 方差、零方差组、实际更新次数、loss/grad norm、tokens、wall time、GPU 小时和保存 step。零方差不冒充有效学习信号。
- [ ] R4.6 从开发 pilot 测量成本，锁定正式预算、评测频次/选点规则及中止条件。两分支使用相同场景顺序/seed、计划更新次数与采样/token 上限，额外重试与实际有效更新另列。
- [ ] R4.7 防误续训与状态污染：独立 run/model identity/workspace；检查源 checkpoint、优化器初始化、reward 版本、数据 hash 与代码版本；不删除原 `.art`。

**交付：**固定 GRPO 配置、隔离策略测试、开发 pilot 与训练日志 schema。**验收：**基础设施失败不形成伪负奖励；组大小/方差/更新可解释；V1/V2 只有预注册奖励差异，其他变量一致。实际预算偏差超出预注册容忍范围时不能称“等预算”，须补跑或限制结论。

### R5：四组主实验与验证集选点

| 实验 ID | 模型路径 | 用途 |
|---|---|---|
| B0 | 原始 Qwen2.5-0.5B-Instruct | 未训练能力基线，不误加载旧 LoRA |
| B1 | Base → SFT | 测量正确工具轨迹监督的作用 |
| G1 | 共享 B1 checkpoint → GRPO V1 | 原奖励训练效果及漏洞 |
| G2 | 同一 B1 checkpoint → GRPO V2 | 轨迹奖励是否改善答案/依赖执行 |

- [ ] R5.1 锁定所有版本/配置后运行四组；B0/B1 也采用同一推理/评测协议，记录各自训练成本。V1/V2 的公平预算比较是 G1 对 G2，不强行声称四组训练总成本相同。
- [ ] R5.2 定期仅评 validation，按预注册最终答案主指标选 checkpoint；例如主指标相同按依赖有效率、再按更早 step 打破平局，规则必须先固定。
- [ ] R5.3 保留 last 与 best，记录选点依据、评测次数与实际训练历史；不用 train reward 或 test 挑模型。
- [ ] R5.4 验证每条 run 实际更新了权重；检查零方差、只调用一次、动态分数始终为 0 等信号，解释什么时候首次出现完整多步轨迹。

**交付：**四组原始验证结果、best/last manifest、训练曲线数据。**验收：**同 SFT 起点、可比预算、无 test 调参；改善必须出现在独立答案/链路指标，不能只比较 reward。若无法出现多步轨迹，先分析专家轨迹/探索/截断/奖励稀疏度，不直接继续堆 steps。

### R6：一项消融、种子与冻结测试

- [ ] R6.1 预注册一项可回答机制问题的消融。优先候选：V2 移除动态依赖奖励，其他分项/超参数不变；记录奖励量级可能随之改变的局限，不能把变化全归因于依赖机制。
- [ ] R6.2 若改选“无 SFT → GRPO V2”，将其作为与 G2 对照的 warm-start 消融；相同 RL 预算并另列 SFT 额外成本。若只做消融而不补齐四组主实验，注明研究范围缩减，不能勾选完整版完成。
- [ ] R6.3 预算允许时完成 2–3 个预定种子；每个 seed 内 G1/G2 使用同一 SFT 起点，区分只重复 RL seed 与重复完整 SFT+RL 链路。资源不足则明确 single-seed。
- [ ] R6.4 在所有配置、模型选择、checker 与报告口径锁定后，一次性批量评冻结 test 的 B0/B1/G1/G2/消融及所有计划 seed；不能先看一组 test 再改另一组。
- [ ] R6.5 保存所有 attempts、退出码、GPU/资源清理与环境记录；按场景配对差值、置信区间和种子波动分析。多种子的均值/标准差与有限测试集不确定性分开报告。

**交付：**消融配置与假设、冻结测试原始结果、seed/配对统计。**验收：**至少一项受控消融；test 不参与选点；无未披露后验改动/丢样本。无需为了“好看”得到正结果。

### R7：形成能讲清楚的简历研究产物

- [ ] R7.1 一张主结果表：四组+消融，答案准确率、答案 Pass@8、分深度链路成功、动态依赖、infra、成本、种子、checkpoint；给分子/分母、区间与缺失项。
- [ ] R7.2 训练/validation 曲线：统一横轴定义，reward、答案/链路、组方差、有效 rollout、实际更新与 tokens/GPU 小时；无训练的 B0 不伪造训练曲线。
- [ ] R7.3 典型轨迹：至少覆盖正确单步、正确多步/依赖（若存在）、只做一步、参数错误、替代合法解、高分错误。保留 raw trace 与 checker 解释，不只挑成功案例。
- [ ] R7.4 简短实验报告：问题、假设、数据划分、算法与 ART 边界、公平控制、结果、消融、失败与局限、复现说明。记录上游来源与自己的改动。
- [ ] R7.5 更新 README/简历介绍的历史状态，新增证据索引；不把不同环境/旧数据/不同 checkpoint 数字拼成提升曲线。
- [ ] R7.6 形成 3 条简历 bullet 和面试问答；所有数字追溯到 run、dataset/checkpoint hash、checker 版本与 raw report。

**验收：**他人可按固定配置复核主要计数与实验身份；所有结论都有证据或明确“不足以判断”；不能复现的外部服务、环境/框架限制如实说明。

## 5. 实验留档与职责分工

拟定产物组织（均待实施，不是现成命令）：

```text
data/research_v1/              # 固定三份数据、数据卡与manifest
configs/research/              # 研究协议与各实验固定配置
results/research/<run-id>/     # 轻量原始结果、退出码、manifest与统计
<Drive>/<run-id>/              # checkpoint、大日志/trace与备份
```

每次 GPU run 至少记录：代码 commit + dirty diff、数据/工具/schema/checker hash、源模型 revision、初始/最终 checkpoint hash、实际配置、seed、GPU/驱动/软件清单、运行命令、有效更新与 rollout、时间/成本、日志、原始评测、退出码、结束后的 GPU 进程查询。

延续现有 [Colab 反馈规范](colab_phase3_validation_zh.md)：SSH 拉取/推送；小型结果放独立 `feedback/<run-id>`，不自动合并；大文件留 Drive，记录大小/SHA-256/位置。不要上传私钥/token、`.art` 或大权重；不要为了新实验删除旧 checkpoint。生命周期操作参考 [已有验证指南](colab_lifecycle_validation_zh.md)，但该指南的旧 2-step smoke 不代替新研究训练流程。

| 角色 | 每一步的职责 |
|---|---|
| 本地助手 | 设计规格、实现与测试、审查配置/数据、提供经核对的运行指南、复核反馈、更新任务书与报告 |
| 用户 | 决定预算/期限/范围；执行需要 GPU 的 Colab 单元格，保管 Drive 权重，推送反馈，确认不可逆或扩预算操作 |
| 共同验收 | 根据证据决定通过/修复/缩减，不把“代码完成”当“GPU实验完成” |

遇到外部兼容性 blocker，先解决最小加载/更新 pilot；不在基础功能未验证时购买/启动大实验。预算、环境未确认前，本任务书不提供臆造的步数、GPU耗时或价格。

## 6. 简历与面试边界

### 当前可以写

“基于 ART 复现 Qwen 小参数模型工具调用 GRPO/LoRA 训练闭环，实现轨迹诊断、评测稳定化与可选奖励改造，并完成 GPU smoke 和后端生命周期修复验收。”

当前不能写：独立泛化准确率提升、V2 训练增益、多步能力显著提升、完成 50 steps、从零实现 GRPO。173 tests 只代表现有版本记录，不预先代表未来研究代码测试数。

### R7 完成后可按实测填写（不得保留占位数字作为成果）

- 构建 **[实际规模]** 个可执行工具任务，按语义实例/参数与泛化分层划分独立数据，建立答案、链路与动态依赖评测。
- 从相同 SFT checkpoint 出发，以 **[实际可比预算]** 对比 V1/V2 GRPO，冻结测试答案准确率由 **[实测 A]** 至 **[实测 B]**，差值 **[百分点与区间]**；报告 **[种子数]** 与局限。
- 用 **[实际消融]** 验证 **[证据支持的机制/未支持的假设]**，分析缺步/参数错误/高分错误；训练与算法执行使用 ART，个人贡献为数据、奖励、诊断、评测和实验设计。

没有正向增益时改写为“完成受控对照，发现……并定位……”，不制造数字。

### 必须能解释的面试问题

- GRPO 的同 prompt 分组、相对优势、标准差与零方差；实际 ART 版本的实现是否与论文公式一致？
- token logprobs、old/current/reference policy、ratio/clipping、KL/beta 在本次配置中是否启用？不要描述未实际使用的项。
- 为什么工具 observation 不是 assistant 的监督目标？多 turn 的 loss mask/截断如何影响学习？
- 稀疏奖励、warm-start 与探索之间有什么关系？V2 是训练改进假设还是已验证机制？
- 为什么同轨迹重评分不能证明训练提升？为什么 reward >2 不等于答案正确？
- 为什么不同 seed 生成数据仍可能泄漏？怎样处理改写、参数和组合泛化？
- 答案正确但参考协议不符该怎么算？匹配数值为什么不能单独证明因果依赖？
- infra 过滤怎样改变组大小/采样偏差？Pass@8、rollout rate、置信区间如何区分？

## 7. 一步步推进方式与当前下一步

每轮固定流程：

1. **确定一个小任务**：输入、修改范围、通过条件、是否需要 GPU。
2. **本地完成并检查**：保持旧数据/结果不变，先运行相关测试，再给用户运行说明。
3. **用户执行必要 GPU 操作**：保留 checkpoint 和原始反馈，不越过预算。
4. **复核证据**：不只看摘要或 reward；核对模型身份、分母、有效更新与异常。
5. **更新本任务书**：只有附证据才勾选；记录下一项与 blocker。

**当前已确认：完整研究闭环，A100 40GB / 20 GPU小时 / 两周；固定合成快照、结构化JSON答案。** 候选train1200/val180/test240不变，R1.6/R2.1–R2.8本地CPU合同通过；R0.3决策预注册与R1.8操作性test冻结已锁定，预算/训练配置附录未就绪。证据：[数据验收](research_dataset_implementation_zh.md)、[干预/checker验收](research_checker_interventions_zh.md)、[本地评测CPU合同](research_evaluation_zh.md)、[真实CPU pilot](research_cpu_pilot_20261012_zh.md)、[预注册/冻结协议](research_preregistration_v1_zh.md)；当前全量506tests通过，模拟诊断不是模型能力。

最新：**单独授权的 Base CPU validation pilot v2 完成真实身份加载/生成及 raw 重放，18/18返回有效、0infra/缺失，75.97s，无GPU。** 固定前两条均为IID二步年龄任务，答案和正确工具链均未成功；仅验证了真实错误工具反馈路径，不能宣布完整R2或多步能力。

最新R3：用户已授权并完成R3.1–R3.3，[专家/SFT数据验收](research_sft_data_zh.md)记录train1200/1200通过、真实tokenizer与assistant-only mask/截断检查、全量532tests通过；无模型加载/推理/训练/GPU，冻结源及旧证据不变。

下一项：本地SFT入口/合同测试已完成，用户按[Colab指南](colab_research_sft_pilot_zh.md)拉取独立分支、核对预算/分配时间、先运行tests/preflight，再单独确认两次更新+Transformers重载pilot。本地不自动启动GPU，ART/vLLM另行兼容pilot仍待授权；在正式训练前锁定预算附录。正确模型多步/checkpoint/目标后端仍待验；不使用test调试，不覆盖冻结源和旧证据。

## 8. 验收日志

| 日期 | 范围 | 证据 | 结论/下一步 |
|---|---|---|---|
| 2026-10-11 | 建立任务书、只读复算旧数据 | 本地 `09e2905`；train/val 哈希与第 2 节一致；200/86、50/37、36 条重叠；既有反馈复核链接 | 任务书已建立；R1–R7未开始。未训练、未修改数据/源码 |
| 2026-10-11 | R0.1 / R0.2 范围与资源确认 | 用户问卷：完整研究闭环；Colab A100 40GB；总计20小时；两周内 | 已记录资源上限；下一步模板审计与数据/评测规格；R0.3/R0.4尚未完成 |
| 2026-10-11 | R1.1审计与数据/评测规格 | 审计脚本/JSON；21新增测试，全量194passed；freshPyright0errors、Ruff通过；用户确认合成快照/JSON答案/1200+180+240目标 | 审计通过；R1.1勾选。新环境、数据、oracle和训练尚未实现；下一项schema/环境/生成器及CPU验收 |
| 2026-10-11 | R0.4 / R1.2–R1.5 / 候选R1.7 | 独立schema/环境/生成器；1620候选+manifest/card/audit；81新增CPUtests，全量275passed；freshPyright0errors、Ruff通过，完整字节重生成一致 | 容量与禁止的跨split泄漏关卡通过，旧数据/工具hash不变；事实复用/实体新颖性单列。R1.6、test冻结、R2 checker和训练未完成 |
| 2026-10-11 | R1.6 / R2.1–R2.4 CPU开发 | [干预/checker验收](research_checker_interventions_zh.md)；独立stdlib oracle、协议/依赖分离；95新增tests、全量370passed；180参考、106配对、4类各106负控、10合法绕过；freshPyright/Ruff通过、诊断与原候选字节重生成一致 | 本轮关卡通过，不是模型准确率或训练增益；原数据/工具不变、无GPU/模型推理/commit/push。R2.5–R2.8、预注册、test冻结与训练仍待做 |
| 2026-10-11 | R2.5–R2.8本地入口/CPU合同 | [评测验收](research_evaluation_zh.md)；显式身份、固定协议、全量attempts、分层/区间/raw重放；100新增tests、全量470passed；fresh九文件Pyright0errors/Ruff通过；15文件--check及HASHSEED99一致 | CPU实现关卡通过：两个模拟身份各72返回/70有效/2infra，比较无效，无补采样。未真实加载/模型推理/GPU/训练/commit/push；旧候选与干预产物一致。真实模型/后端pilot、R0.3、R1.8与R3仍待做 |
| 2026-10-12 | 单独授权 validation Base CPU 真实加载/推理 pilot v1 | [CPU pilot 记录](research_cpu_pilot_20261012_zh.md)；官方 revision `7ae557604adf67be50417f59c2c2f167def9a775`；独立 Python 3.12 / torch2.8 / transformers4.57.1；失败 run 与外层时间/exit/log 留档 | exit=1：未启用 sliding_window 的运行时配置规范化触发严格核验。2 场景 / 18 planned，0 returned / 0 valid / 18 missing；未完成加载身份验收或生成，未重试/补样，GPU0。待窄范围兼容修复及新 run 复验；不勾选新关卡，不是模型能力成绩 |
| 2026-10-12 | 用户另行授权窄配置兼容修复 + Base CPU pilot v2 | [真实CPU pilot](research_cpu_pilot_20261012_zh.md)；限定未启用窗口的32768→null规范化；10新增/480passed，freshPyright/Ruff通过；独立新fixture字节重生成与v2生产audit通过，v1失败原样保留 | v2真实加载回执CPU float32/无adapter；18planned/returned/valid，0infra/缺失；36生成请求，75.97s，exit0/audit0、PID退出。greedy0/2、sample0/16、Pass@8为0/2，18条模型工具错误保留；无正确多步证据、无GPU/训练/test推理/commit/push，不是泛化对照。原数据/干预不变；旧模拟manifest源码hash变化另存新证据、不覆盖旧记录 |

| 2026-10-12 | R0.3决策预注册 + R1.8操作性test冻结 | [协议](research_preregistration_v1_zh.md)、[验收报告](../results/research/registration_v1_20261012/REPORT.md)；用户确认五臂/移除动态奖励/seed20261011/选点/失败策略与独立副本；原六文件bytewise相同、testhash不变，20份来源hash封存，26新增/506passed7.62s、freshPyright/Ruff通过 | create/check/HASHSEED99通过；原数据/干预--check与既有CPUfixture/真实pilot只读audit通过。R0.3/R1.8勾选仅决策协议/数据纪律，formal_training/final_test就绪均false。无本轮GPU/训练/模型推理/commit/push；预算与有效训练/奖励配置附录待成本pilot，不宣称盲测或完整研究完成 |

| 2026-10-12 | 用户授权R3.1–R3.3 train-only专家/独立检查/SFT预处理 | [验收报告](research_sft_data_zh.md)；冻结train1200重新执行并独立校验，接受1200/拒绝0；深度360/480/360，真实固定revision tokenizer739–969tokens、assistant监督120963tokens；26新增、532passed9.94s、freshPyright0errors/Ruff通过；两次--check（含HASHSEED99）完整字节一致、冻结登记--check通过 | R3.1–R3.3勾选仅数据/CPU/tokenizer关卡；默认环境缺psutil首次收集失败已披露，隔离依赖全量通过。无模型权重加载/推理/训练/GPU/test调试/commit/push；R3.4–R3.6及正式预算/兼容仍待验，不宣称warm-start收益 |

| 2026-10-12 | 用户授权本地SFT入口/合同与远程供Colab拉取 | [入口验收](research_sft_entry_zh.md)、[命令指南](colab_research_sft_pilot_zh.md)；固定3条train、2次更新、loss/mask/grad/逐次权重变化/save/fresh reload合同；专项61passed、全567passed11.21s、fresh六文件Pyright0errors/Ruff通过；tokenizer-only preflight exit0、旧SFT与冻结字节检查通过 | R3.4仅本地入口部分，不勾选；本地只随机微型CPU模型合同，无官方权重训练/GPU/test推理。用户服务器须核对剩余预算/分配时间并单独确认A100最小pilot；ART/vLLM与正式训练仍未授权/验收。独立分支推送是否成功以实际git核对为准，未改远程master |

后续只追加已完成事实与证据；本任务书初版不是授权无限制实现、训练或消费 GPU。

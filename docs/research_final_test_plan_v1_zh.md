# 四模型独立最终test：冻结附录 v1

状态：**仅本地准备，GPU执行未授权、test未运行。** 本文替代后续继续追validation的建议，不替换历史预注册/锁。旧train/validation入口仍拒绝test；新增独立入口 `python -m scripts.eval_research_final_test_v1`，默认仅核对字节和240行计数，不解析test题目/标签、不加载模型。

## 冻结对象与主要比较

机器配置：`configs/research/final_test_v1.json`；代码/数据/配置绑定：`configs/research/final_test_v1.lock.json`。四模型固定顺序如下，路径相对Drive `MyDrive/`：

| 模型 | 固定模型（不加载其AdamW） | 身份fingerprint前缀 |
|---|---|---|
| SFT30 | `AgenticRL-research-sft-to30/20261018T013743Z/run/best/checkpoint-30` | `467e2a3253af` |
| seed20261011 | `AgenticRL-answer-grpo-v1/20261018T071612Z/run/seed-20261011/batch-0060/checkpoint-28` | `9299d39de2ae` |
| seed20261012 | 同campaign `seed-20261012/batch-0060/checkpoint-31` | `f049ed03779a` |
| seed20261013 | 同campaign `seed-20261013/batch-0060/checkpoint-31` | `9f6806f56482` |

完整指纹从原始两个反馈ZIP的selection元数据取得，不是本地独立加载模型的证明。SFT历史selection指向stage-30，最终使用best副本，运行前必须以同一完整指纹核对。四个身份全部通过后才读取test，不能换候选、只评最高seed或用文件名代替hash。

主要估计量为 **三个GRPO种子答案正确率均值 − 同一SFT30正确率**。四组各240道相同冻结题，合计960个任务，而非960个生成请求。test SHA256固定为 `08040a4e2e7b7c410f17937496a22bccd09f925e4c5dba35da2f027380410a52`，不重新抽题。

## 协议不变

复用 `SFTValidationBackend._generate` / 既有Base+PEFT加载器和 `evaluate_one` / 原独立checker、工具执行器及prompt。Qwen2.5-0.5B-Instruct固定revision，A100、torch2.8.0 / transformers4.57.1 / peft0.17.1 / safetensors0.8.0，Base BF16/eager、adapter按原加载精度；greedy、单并发、6turn、每turn512输出tokens、总输出2048、prompt4096、单请求60s。tokenizer/template/schema/源代码一并hash绑定。四模型顺序加载，eval+inference_mode，无优化器、反向传播或训练状态恢复。

旧协议文件的 `test_policy=development-only-reject-test` 原样保留，它只授权开发入口；新最终附录授权范围单独记录在manifest的binding中，不能将它误读为旧协议已自动授权test。正式运行还需确认余额、明确硬预算和执行口令。

## 保存、失败和中断

每个任务保存一个独立JSON：原始响应/token IDs、每次请求/messages、工具事件/observation、checker、错误分类、tokens、耗时、CUDA峰值allocated/reserved。原子写本地、fsync、复制Drive临时文件再提交，随后**重新打开两边核hash并解析核对**，成功后才能下一个任务。结束重新读完每组240份文件并通过原raw响应精确重放checker，才生成完整比较。Drive保存失败立即停止。

manifest在请求之前先持久化running状态。OOM/timeout/基础设施错误保留原失败任务并停止整个cohort，不自动重试/跳过；不调整上下文、生成上限、prompt、奖励或判分容差。外部硬中断可能留有未收到的任务，必须按未完成披露，不可凭空补行。

只支持 `paused-clean-boundary` 的显式 `--resume`：同一个目录、同样四个身份/源代码/配置，重新读取并重放已保存连续前缀，然后继续缺失后缀，已完成题不重复计数。failed/running/loading/未知中断一律拒绝自动续评，先回传核查，不另开目录悄悄重跑。若Colab本地盘消失，先完整还原原Drive目录后核对；不要删原证据。第一次触及test即记test_consumed，不能因失败再把它称作从未使用。

## 报表与统计事先约定

每组报告答案n/240、strict、semantic、动态依赖实际分母、平均工具调用、请求、生成/输入tokens、任务耗时和会话总耗时。分family/depth/generalization、unit_error/parse_error等错误，重点检查三步、组合以及country_density_convert，不只展示受益家族。所有变对/变错题ID保留。

三个GRPO率报告均值与**样本标准差ddof=1**。每seed与SFT做同题配对，变对/变错及差值区间；主要均值差按每题 `(seed11+seed12+seed13)/3 - SFT` 形成240个值，沿用固定seed20261011的1000次场景配对percentile bootstrap（95%区间）。同题四个模型一起重采样，不包装成720个独立题。区间条件于这四个固定模型，只3个seed且SFT只有一个seed，不能证明RL训练方差小于SFT；分层/每seed区间是探索性分析、不宣称多重比较显著性。

代码生成所有分层与配对计数；拿回反馈后本地复算原始结果、补主表/分层和成本解读，才下最终结论。提升消失也接受；不再用此test调参。保留4条训练轨迹缺失、采样/teacher LP差异、非外部盲测等局限。

## 成本：有依据的粗估，不当余额或承诺

原 `results/baseline-feedback.zip` 的B1(step10)完整180validation的manifest记 `wall_seconds=1214.84623581`，含加载/收尾；B0为472.971348693s。若机械按B1任务平均外推，960任务约 **6479s=1.80h**。它不是SFT30/GRPO、更不是test耗时预测：test构成、长轨迹与Drive逐题同步都不同。三seedvalidation manifest没有独立wall_seconds，不能从campaign 5h17m8s拆出精准评测速度，也不能把旧预留2小时当可用余额。

可供余额确认后的讨论起点：纯评测估计上另留约30%（约2.34h），再考虑安装/下载、四次加载、备份与异常空间；**当前不设已获授权的GPU小时数**，不保证2小时跑完。必须取得当前实际余额，按更保守预算确定wall-seconds和外部硬timeout；余额不足就不启动。入口预留420秒停止新任务，外部硬限包括进程启动/预检/加载/评测/保存；安装和闲置另计。

无反向传播通常显存较省，但训练峰5.65GiB仅来自约1000-token轨迹，不能推断4096-token test安全。每题记录实测显存，OOM即失败停止，不缩短协议。

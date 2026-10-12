# R3.1–R3.3：train-only 专家轨迹与 SFT 数据验收

2026-10-12，本地 CPU / tokenizer-only；未加载模型权重、未推理、未训练、GPU 0。R3.4–R3.6 不在本轮范围，不能据此宣称 SFT 更新、warm-start 收益或 ART/vLLM 兼容。

## 实现与产物

- `src/research_sft.py`：真实执行专家、独立检查、精确模板 tokenization、assistant-only labels、右侧 padding。
- `scripts/build_research_sft.py`：固定读取 train，无 split/test 参数；本地 tokenizer，无网络下载/权重加载；独立新目录输出，已存在时拒绝覆盖；`--check` 只读重生成比较。
- `tests/test_research_sft.py`：26 个新增用例，含真实固定 revision tokenizer 检查（仅在该本地环境不存在时显式 skip）。本次无 skip。
- `data/research_sft_v1/experts.jsonl`：1200 条多轮消息、实际 observation/trace、独立 checker 结果与来源 ID。隐藏字段仅用于审计，不进入 messages。
- `data/research_sft_v1/tokens.jsonl`：input_ids、attention_mask、未移位 labels、assistant token ranges、长度统计。约 12.2 MB，无权重。
- `data/research_sft_v1/manifest.json`：完整来源 sample IDs、train/manifest hash、tokenizer 文件与模板 hash、实现 hash、接受/拒绝台账、长度与监督统计。

## R3.1 / R3.2

来源为 `data/research_v1_frozen/train.jsonl`，SHA-256 `b98fc61cfae88426e065ff817db9757650944c76f3268298de82eb76a097048e`。只载入并解析 train；不载入 validation/test 记录。冻结校验命令会校验 test 原字节/hash，但不解析标签。

从 recipe、公示参数与不可变环境重新调用已执行的参考构建器 `execute_reference`。每次工具执行完成、解析实际 observation 后，才构建下一 turn 的动态参数；最终答案同样取自执行结果。不复制保存的 reference_trace、expected_calls 或 oracle_answer，不填 `__dynamic__`。每调用单独 assistant turn，紧跟带同一 call ID 的 tool response，再进入后续 turn。

每条重建 public system/user + 工具消息，与保存 messages 逐项比较，拒绝隐藏 prompt 元数据、call ID/trace 错配、同 turn 假依赖。由 R2 的独立 stdlib oracle 判答案，再由独立 route builder/replay 判真实执行、strict、semantic、dynamic；不导入 reward，也不信任轨迹的 success/checks 字段。

结果：来源 **1200**，接受 **1200**，排除 **0**。答案、执行、严格链路、语义路线均通过；有动态边时 dynamic=valid，无边为 not_applicable。深度 1/2/3 = **360/480/360**。逐条 checks 与拒绝原因字段可复核；有错误时不静默丢弃，而在 summary.rejected 中逐 ID 留账。源文件 schema/身份失效时整次拒绝。

局限：专家构建复用数据生成器；执行/协议检查共享环境实现，仍存在共同错误风险。独立答案 oracle 不使用工具/参考 gold，并有 R2 的手算对照。所有轨迹通过不是模型能力成绩。

## R3.3 模板、序列化、mask 和截断

使用官方 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775` 已有本地 snapshot，Transformers 4.57.1 的 fast tokenizer；不覆盖原 chat_template。模板 hash：`2e2d2512cfe46af53dc1eed45368ecaab26ac4461c480e6b69fe571cf0ceaa75`。工具 schema 与 system prompt 同 R2。

Qwen 原模板把 assistant tool_calls 序列化为 `<tool_call>` name/arguments JSON，并把 observation 包进 user-role `<tool_response>`。**不是凭序列化后的 user/assistant 字样推断 mask**：按原 messages 中的 assistant turn，用 prefix render 定位 payload 到 `<|im_end|>` 的字符区间，再在完整一次 tokenization 的 offsets 上对齐。断言前缀稳定、边界不跨 token、与原生 apply_chat_template(tokenize=True) 完全一致；不依赖该模板没有提供的 generation mask，也不对整条输入做字符串搜索来猜监督范围。

- 监督：每个 assistant 工具调用 payload、最终 JSON 答案、每个 assistant 的 EOS `<|im_end|>`。
- 不监督：system prompt、tool schema、user task、工具 observation/response wrapper、assistant role header、EOS 后的分隔 newline；labels=-100。
- 每条必须保留 depth+1 个完整 assistant span；每 span 的末 token 必须是 EOS。
- labels 与 input_ids 对齐、**不预先 shift**，后续 causal-LM 实现负责 shift。
- 上限 4096 仅为本轮数据预处理合同，不是正式训练预算。超过上限时整条拒绝并记录实际长度，不左截/右截/部分保留；等于实际长度允许通过。
- 无 packing；padding 在 collator 中右补，attention=0、labels=-100；不使用会重新生成全序列 labels 的通用 LM collator。训练入口接入这些约束仍需 R3.4/R3.5 验证。

长度 **739–969 tokens**，总输入 **1,049,899**，assistant 监督 **120,963**；全部低于 4096，截断/长度排除 0。真实 tokenizer 测试覆盖单步/二步/三步、独立比较、多轮 tool observation 全部被 mask、每 turn EOS、精确长度允许/少 1 token 拒绝以及 padding。测试还覆盖非 train 提前拒绝、伪 observation/参数、缺步、同 turn、错答案、隐藏 prompt、call ID 错配、gold 毒化与重复组。

## 复现与验收

```bash
PY=/Users/wafoe/.cache/agenticrl-pilot-venv/bin/python
TOK=/Users/wafoe/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
# 已有产物只读复核；首次构建去掉 --check，并选不存在的新目录
$PY scripts/build_research_sft.py --tokenizer-dir "$TOK" --output data/research_sft_v1 --check
PYTHONHASHSEED=99 $PY scripts/build_research_sft.py --tokenizer-dir "$TOK" --output data/research_sft_v1 --check
python scripts/freeze_research.py --check
uv run --with pytest --with pytest-asyncio --with psutil --with pyyaml pytest -q
uv run --with pyright pyright src/research_sft.py scripts/build_research_sft.py tests/test_research_sft.py
```

两个 `--check` 均完整字节一致；冻结登记校验通过，formal_training_ready/final_test_ready 仍 false。产物含本地绝对来源路径及实现 hash，换环境复现内容时应在新目录保留新的 provenance，不覆盖旧 manifest。

- experts SHA-256：`b7b7f3129517941ef32316ae83779769b9ac065ffe66e6367b1b7dbcc342fa92`。
- tokens SHA-256：`90da76311716725362827d89179a1fa1d6a908de8137c3da9b7a5acb0e7ba19d`。
- 专项 **26 passed**；全量 **532 passed / 9.94s**。
- 默认 Python 的首次全量收集因缺 psutil 失败，未伪称通过；随后使用隔离 uv 的 pytest/pytest-asyncio/psutil/pyyaml 依赖完整通过，未安装 ART/GPU 栈。
- fresh Pyright 三文件 **0 errors / 0 warnings**；Ruff E/F/I/B（忽略 E501）通过。
- 活跃 LSP 对新模块两处 import 报 stale resolution，经运行时 import 与 fresh Pyright 验证后标为 false-positive；EOS 被误识别为密码也标 false-positive，不写 suppression、不放宽代码检查。

本轮未修改旧数据、冻结源、R2 checker、预注册锁或历史结果；无 commit/push。下一项须另行授权 R3.4：SFT 路径/真实更新与 checkpoint / ART-vLLM 兼容 pilot；正式训练前仍须锁预算附录。

# Validation 真实 CPU pilot（2026-10-12）

## 运行前固定范围

用户单独授权少量 validation 真实加载/推理，并选择本地 CPU、允许安装依赖和下载官方固定 revision 权重。

- 模型：原始 `Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775`（本次从官方 Hub API 查询后固定）。不加载项目 LoRA。
- 环境：独立 `/Users/wafoe/.cache/agenticrl-pilot-venv`，Python 3.12，torch 2.8.0 / transformers 4.57.1 / peft 0.17.1；CPU float32，不启用 MPS/CUDA。
- 权重：官方 HF 本地 snapshot；当前源码的 identity 合同允许 snapshot symlink 指向同一 repo 的 blobs，实际目标字节纳入 SHA-256。不是额外伪造来源元数据。
- 数据：`data/research_v1_candidate`，validation 文件固定前 2 条；不按答案或成功情况挑选。不读 test 做推理。
- 原协议：`configs/research/evaluation_v1.json`；每场景 greedy 1 + sample 8，总计 18 planned attempts。6 turns，512/2048 输出 token、4096 输入 token、60s 请求软限、并发 1，保持不变。
- 只跑一个新 run ID：`base_cpu_pilot_20261012_v1`。不自动重试、补采样、追加、扩大数据或调整预算。失败保留原始计划、ledger 与错误；必要修复/重跑先说明。
- 验收：真实加载回执/模型哈希、请求响应和 token IDs、真实工具执行/observation、完整分母、结束状态及原始结果 audit 重放。若模型未答对，如实报告，不作为基础设施失败过滤。
- 本次不训练、不进行正式模型比较、不消耗 GPU；不能证明 PEFT checkpoint、ART/vLLM 或 GPU 兼容，不能宣称完整 R2 或 R3 已通过。两场景不是代表性 benchmark。
- 60s 为生成软限，CPU forward/取消收尾可能超时；不保证耗时。完整过程以外层资源/时间/退出码记录为准。

## 执行命令

```bash
PY=/Users/wafoe/.cache/agenticrl-pilot-venv/bin/python
BASE=/Users/wafoe/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
$PY scripts/eval_research.py --base-dir "$BASE" \
  --base-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --split validation --limit 2 \
  --output results/research/base_cpu_pilot_20261012_v1 --run-cpu
$PY scripts/eval_research.py --audit --output results/research/base_cpu_pilot_20261012_v1
```

## 结果

**v1 失败，未完成真实推理验收。** 运行日志位于 `results/research/base_cpu_pilot_20261012_execution/`，原始失败 run 位于 `results/research/base_cpu_pilot_20261012_v1/`；未覆盖、补采样或重跑。

- 外层推理命令 exit=1，wall time 11.47s；GPU 用量 0。异常：`ValueError: loaded model config differs from actual local config`。
- 执行已进入真实 Transformers 权重加载路径，但加载回执尚未通过、manifest 中 `loader_receipt=null`；不能宣称身份验收成功。
- 计划 2 个场景、18 attempts；实际返回 0、有效 0、未返回 18（greedy 2 + sample 16）。加载失败是 run 级基础设施错误；ledger 未返回项不能当成模型能力错误。无模型准确率、无有效比较。
- 只读 `AutoConfig.from_pretrained(..., local_files_only=True)` 复核：官方 config 的 `sliding_window=32768` 被 Transformers 4.57.1 在 `use_sliding_window=false` 时规范化为 `null`；其余受检查字段一致。该诊断没有重新加载模型权重或执行推理。
- 未执行成功 run 的 `--audit`，因为失败 run 不满足入口要求；原始失败 artifacts 仍完整留档。
- v1 之后用户另行确认：窄范围兼容修复、补回归测试，并以新 run ID 原样重跑 2 个 validation / 18 attempts。v1 不覆盖、不追加，不把两次计划混成一次成功运行。不能通过无条件忽略 sliding_window 或放宽全部 config 检查来规避。
- 此结果不足以证明 CPU 太慢或需要 GPU；问题首先是配置比较兼容性。LoRA、ART/vLLM、GPU 兼容均未验证。

## 用户另行授权的修复与 v2 复验

源码只修改 `src/research_eval_backend.py` 的配置比较：限定 Qwen2、file/runtime 两侧显式布尔 `use_sliding_window=false`、文件整数窗口恰为 32768、运行时字段存在且为 null。其余字段保持检查；错误信息列出差异字段。未修改数据、prompt、采样、checker、权重或 reward。

新增 10 个回归用例：接受明确规范化与原值不变；拒绝窗口启用、运行时启用、字段缺失、其他窗口值、其他配置字段变化、缺少开关以及非布尔开关。全量 **480 passed (7.23s)**，fresh 两文件 Pyright 0 errors/0 warnings，Ruff E/F/I/B（忽略 E501）通过。编辑器 LSP 无新增报告，但隐藏了 2 项既有 disposition，不把隐藏诊断当完全无发现。

原候选数据和干预 artifacts `--check` 字节一致。旧模拟评测产物的 `--check` 在源码变更后不再字节一致：与新产物比较仅 Base/checkpoint manifest 的 `source` 及外层 checksum manifest 改变；未覆盖旧产物。新独立 `results/research/evaluation_cpu_pilot_fix_v2/` 合同重放及完整字节 `--check` 通过。

v2 命令与上文相同，输出改为 `results/research/base_cpu_pilot_20261012_v2`。外层记录在 `results/research/base_cpu_pilot_20261012_execution_v2/`：

- 真实加载回执：`Qwen2ForCausalLM`、CPU float32、`real_model_loaded=true`、active adapters 空；模型文件运行前后身份检查通过。
- 推理 exit=0，75.97s；生产 `--audit` exit=0，原始响应/执行/provenance/checker/统计及 artifacts 重放通过。无残留 wrapper/inference PID。
- 2 场景均为固定文件前两条 age_calculation / IID / 参考深度 2，不能宣称覆盖所有深度和泛化层。
- v2 计划/返回/有效 18/18/18，infra 0、缺失 0；36 次真实生成请求，1726 completion tokens、26324 prompt tokens（多轮累计）。
- greedy 答案 0/2，sample 答案 0/16，观测答案 Pass@8 0/2；strict/semantic/dynamic 均未成功。`comparison_valid=true` 只表示台账完整，当前没有第二模型或正式对照。
- 18 条均含工具执行错误，错误 observation 为 13 次 unsupported conversion、4 次 invalid tool/arguments、1 次不支持的算术。greedy 均尝试 years→years 的 convert_units 后未给合法最终答案。真实错误 observation 被返回给模型；不把模型无效参数当 infra，不删失败。
- 日志仍有 `torch_dtype` 弃用警告；不是零告警。未为消除警告扩大本次修复范围。
- GPU 用量 0，无训练/test 模型推理/commit/push。Base CPU 身份加载、生成和错误工具反馈路径已通过；没有正确工具轨迹或动态依赖成功证据，PEFT checkpoint/ART-vLLM/GPU 仍待验。

汇总：[v2 pilot 报告](../results/research/base_cpu_pilot_20261012_execution_v2/PILOT_REPORT.md)。下一步先补齐 R0.3 预注册与 R1.8 冻结纪律，并单独安排正确多步/真实 checkpoint/目标后端兼容的验证；不得从两条失败案例推导泛化能力或训练增益，也不自动进入 SFT/GRPO。

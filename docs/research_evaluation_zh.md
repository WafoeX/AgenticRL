# R2.5–R2.8：本地评测入口与 CPU 验收

## 1. 结论与边界

已实现独立研究评测入口，并完成 **100 项新增 CPU 测试，全量 470 passed（7.72s）**；九文件 fresh Pyright 为 0 errors / 0 warnings，Ruff `E,F,I,B`（忽略 E501）通过。

**这是实现/模拟响应合同验收，不是真实 Base/checkpoint 的模型成绩。** 本轮未下载或加载真实权重、未启动 GPU/ART/vLLM 服务、未训练、未用 test 做模型推理、未 commit/push。旧工具、旧数据和既有候选数据、干预产物的字节重生成检查通过。

- R2.5–R2.8 的本地实现与 CPU 合同已验收；真实 Qwen/PEFT 加载、生成以及 ART/vLLM 兼容仍待小规模 validation pilot。
- Base 指未加本项目 LoRA 的 `Qwen/Qwen2.5-0.5B-Instruct`，不是另一个预训练版模型。
- R0.3 正式预注册、R1.8 test 冻结、SFT/GRPO 尚未完成；不据此宣布完整研究或训练关卡通过。

## 2. 实现入口

| 文件 | 职责 |
|---|---|
| `configs/research/evaluation_v1.json` | 固定评测协议，独立于 reward/训练配置 |
| `src/research_model_identity.py` | 显式本地 Base/LoRA 身份、revision/step、文件 SHA-256、加载回执校验 |
| `src/research_eval_protocol.py` | 开发数据 manifest 检查、完整计划、逐请求稳定 seed、协议 hash |
| `src/research_eval_backend.py` | 延迟导入的 CPU float32 Transformers/PEFT 适配器、Qwen 文本工具调用 codec |
| `src/research_evaluation.py` | 多轮真实工具执行、原始响应/call ID/turn provenance、失败保留与关闭 |
| `src/research_eval_metrics.py` | 全量/有效分母、分层 raw counts、准确率、观测 Pass@8、区间及配对差值 |
| `scripts/eval_research.py` | 默认 preflight；显式 `--run-cpu` 才加载真实本地模型；`--audit` 只读重放 |
| `scripts/verify_research_eval_cpu.py` | stdlib 模拟验收与留档/字节重生成，不导入真实模型库 |
| `tests/test_research_evaluation.py`、`tests/test_research_eval_backend_cpu.py` | 入口、台账、重放、篡改、统计、模拟库加载及重复取消测试 |

旧 `scripts/eval_checkpoint.py`、训练/rollout/reward 链路未接入或替换；新入口不以 `reward > 2` 定义成功。

## 3. 模型身份与加载纪律

- 不猜“当前 checkpoint”，不访问默认 `.art`。Base 与 checkpoint 均须显式本地路径；Base revision 必须为固定 40 位小写 hex，而非 `main/latest`。
- 记录实际路径、文件清单/大小/SHA-256、Base/config/template 指纹、checkpoint step/指纹、活跃 adapter、运行时 config hash、模型类、库版本、device/dtype。运行前后复核文件不变；回执与指定身份不一致即失败。
- 限定 Qwen2 decoder-only 与 safetensors（支持严格索引的 shards）；拒绝 pickle、自定义 Python 代码、额外/不明确权重、文件 symlink，以及隐式 quantization config。HF cache 常用 symlink，因此该入口需要**已准备好的实体文件副本**，不会自动整理/下载缓存。
- 非 `snapshots/<revision>` 布局须有 `research_model_origin.json`：`{"model_id":"Qwen/Qwen2.5-0.5B-Instruct","revision":"<固定40位hex>"}`。该标签是来源元数据，**不是 Hub 签名认证**；本地 hash 证明本次具体字节身份。
- LoRA 须显式 step，目录 step/可选 `trainer_state.json` 一致；`research_checkpoint.json` 精确绑定 `step/base_fingerprint/base_revision`。adapter config 的 Base/revision 一致，不能更改 tokenizer/template。旧 checkpoint 未必符合该契约，不自动伪造绑定。
- 真实加载器只读、`local_files_only=True`、`trust_remote_code=False`、CPU context/float32/eager attention，不自动量化；明确设置 adapter、eval、冻结参数并核对设备。CPU RNG 单独设置，不调用 CUDA seed API。
- 模拟库测试验证实际传入加载/生成参数、错误 config/template/device/dtype/旧 adapter 拒绝和线程关闭；它们**不能证明真实 safetensors 可加载或后端兼容**。

## 4. 固定推理协议

版本：`research-eval-v1`。模型仅收到固定 system 协议、公开 task、工具 schema 和实际执行后 observation；完整 record、隐藏世界、gold/reference 不发送。

| 项目 | v1 默认值 |
|---|---|
| 模式/计划 | 每场景 1 deterministic + 8 sample |
| deterministic | greedy，temperature=0、top_p=1；不宣称跨环境字节确定性 |
| sample | temperature=1.2、top_p=.95；CPU适配器 top_k=0 |
| seed | 20261011，按场景/mode/index/turn 派生，不依赖完成顺序 |
| assistant turns / 每 turn calls | 6 / 8；同回复相依调用不能伪装为后续 turn |
| 每 turn / 每 attempt 输出预算 | 512 / 2048 tokens |
| 输入预算 | 每请求 4096 tokens；超出不静默截 prompt |
| 请求超时 / 并发 | 60s / 1；真实 CPU 适配器拒绝并发>1 |
| template / codec | 精确本地 tokenizer template，无隐式 override；`qwen-tool-tags-v1` |
| split / 失败 | 仅 train/validation；保留全部、不自动补样/重试 |

每次生成构造显式 GenerationConfig，不继承 checkpoint 的 sampling penalties。输出保存 token IDs 和 usage，仅移除末尾 transport EOS；保留其他特殊 token，不修复 JSON。文本解析生成的 call ID 标为 `adapter-generated-from-text-not-provider`，不冒称原生服务 ID。原始 request/response、规范化消息、工具 call index/ID、assistant turn 与 observation 来源同时留档。

60s 是请求超时/生成软限，不是任意 forward 的硬中断保证。取消时等待生成线程退出，即使重复取消也不先释放仍在使用的模型；可能超过超时。真实硬挂死/驱动行为仍须 pilot 检查。此评测适配器不采集训练所需 token logprobs，不代替 R4.2 的 ART 原始 Choice/mask 集成。

## 5. 分母、错误与统计

- 输入拒绝重复 sample/semantic group、混合 split、test 和隐式 limit clamp。先保存完整 `plan.json`，每个 planned attempt 有稳定 ID；已返回结果逐条写入并 flush/fsync。
- `attempts.jsonl` 保存真实返回/尝试结果；`ledger.jsonl` 补全未返回项。加载失败、取消和关闭失败也保留计划、缺失项和失败 manifest；不把失败 run 当完成。
- **全量分母**始终是指定计划；**有效分母**要求非 infra/缺失且有真实 assistant 回复。错误工具/参数、单位错、协议错、截断等模型错误不因为成绩差被过滤。实际返回空字符串/空 assistant Choice 是模型失败，保留有效 attempt；空 `choices` 或缺失 transport/usage 才属于基础设施合同失败。
- 任一 infra 或真正缺失使比较无效；仍输出原始保守全量计数。需要按预注册规则另建 run ID 整次重跑，禁止在原 run 追加成功样本。未正式预注册，不擅自重跑。
- 最终答案、strict、semantic、reference dynamic 分开；直接正确答案获答案信用，合法 Fahrenheit 路径单列 bypass。动态数值/时序匹配不等于模型内部因果注意。
- 输出 family、reference depth、dynamic dependency、generalization 四类分层；包含计划/返回/有效/infra/缺失数、答案/链路分子分母、多标签及互斥主错误、已知 tokens。
- 观测 Pass@8 是 **至少一次成功场景 / 全部指定场景**，不是 8 条轨迹平均；没有使用有限 n 的组合估计公式。
- deterministic/Pass@8 使用场景 Wilson 95% 区间；sample rollout 均值按语义场景 percentile bootstrap，固定 seed、1000 replicates；配对差值同样按共同场景聚类。小分层区间宽，single-seed 不代表训练随机性稳定。
- `--audit` 从保存的原始响应重新执行工具/重建 provenance、独立 checker 与指标；校验源码、协议、数据、模型文件和全部 artifact 字节/版本/分母。不是仅重算保存的 success 字段。hash/重放不是数字签名，不能证明被整体协同伪造的日志来自真实服务。

## 6. CPU 留档证据

目录：`results/research/evaluation_cpu_v1/`，共 15 个文件；顶层 `manifest.json` 对导出文件给出 hash/字节数。

| 模拟身份 | validation 案例 | planned / returned | valid | infra | 比较有效？ |
|---|---:|---:|---:|---:|---|
| Base fixture | 8 | 72 / 72 | 70 | 2 | 否 |
| step7 checkpoint fixture | 同 8 | 72 / 72 | 70 | 2 | 否 |

案例覆盖 1/2/3-call 和四泛化分层，注入错误单位但协议完整、错误工具但答案正确、同 turn 伪依赖、长度截断、重复参数键、直接正确答案/合法 Fahrenheit 和空 choices infra。两身份在临时原始目录均通过生产 `audit_saved_run()`；无追加/删除 attempts。

临时假 safetensors 只是哈希合同占位字节，未被模型加载；回执均 `real_model_loaded=false`。导出将临时路径替换成 `<EPHEMERAL-CPU-FIXTURE-NOT-MODEL>`，随后删除临时 fixtures。**内层 run manifests 保存导出前原始临时 run 的校验信息，不是脱敏后 dump 的字节校验；以顶层 manifest/`--check` 校验导出。** 不能对这些已脱敏、权重已删除的 fixture 目录执行生产 `--audit`，更不能称其真实模型加载证据。固定源码/同一工作区下，15 文件重生成以及 `PYTHONHASHSEED=99` 检查均通过。

复现（不会启动模型/GPU）：

```bash
python3 scripts/verify_research_eval_cpu.py --check
PYTHONHASHSEED=99 python3 scripts/verify_research_eval_cpu.py --check
python3 scripts/diagnose_research.py --check
python3 scripts/generate_research_dataset.py --check
uv run --no-project --with pytest --with pytest-asyncio --with psutil python -m pytest tests -q
```

新留档使用 `--output results/research/<new-id>`；已存在目录拒绝覆盖/append。模型源文件改变后，生产 audit 会拒绝，不能用不同字节替换原身份。

## 7. 后续使用与下一关卡

仅在模型资产/身份已经准备并核对后进行 **preflight**；以下变量需要填写，不是现成权重路径：

```bash
python3 scripts/eval_research.py \
  --base-dir "$BASE_DIR" --base-revision "$PINNED_BASE_REVISION" \
  --split validation --limit 8 --output results/research/base_preflight_v1
```

指定 LoRA 额外添加 `--model-kind checkpoint --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-step <实际step>`，必须具备上述绑定元数据。preflight 不加载真实模型、无模型成绩。

后续获授权才用**新 run ID**与 `--run-cpu` 做真实 CPU 推理；`--audit --output results/research/<真实run-id>` 只读重放，需原模型/数据/源码仍可核对。本轮未执行这些真实模型命令。当前 CLI 没有 GPU 或冻结 test 模式；不能把参数自行改成 test 或 GPU 冒充已验收入口。

下一项是小规模 validation 真实模型/PEFT加载与生成 pilot，若转 GPU 必须单独授权并计入 20 GPU小时；ART/vLLM 需另行适配并复验同一协议。随后补齐 R0.3 预注册、R1.8 冻结纪律，再进入 R3 的 train-only 专家轨迹与 SFT 兼容验证。不直接扩大训练或据模拟百分比写能力提升。

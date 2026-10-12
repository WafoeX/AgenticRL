# SFT 本地入口与合同验收（2026-10-12）

## 范围

用户授权本地实现入口/合同测试、推送供服务器拉取；Colab A100 pilot 由用户按[独立运行指南](colab_research_sft_pilot_zh.md)先检查再单独确认。本地没有加载官方权重训练、没有 GPU run、没有模型 test 推理或正式 SFT/GRPO。**R3.4 只完成本地入口部分，保持未勾选。**

文件：`src/research_sft_pilot.py`、`scripts/pilot_research_sft.py`、`configs/research/sft_pilot_v1.json`、`requirements-sft-pilot.txt`、`tests/test_research_sft_pilot.py`；已有 SFT tokenizer 测试新增远端 Python/tokenizer 显式路径与非 train 提前拒绝测试。

## 合同

- 不提供更新次数、split、LoRA 或正式训练超参覆盖参数；配置只能等同有版本的两步 v1 合同。独立新 output 目录、拒绝复用、不得位于源码仓库内。
- 默认 preflight：验证原冻结登记与来源；只解析 train，取文件顺序第一个 depth1/depth2/depth3，共3条；真实执行专家并独立校验、重新 tokenization 与 mask，不信任历史 token 文件或保存的 success 字段。仅 tokenizer，不加载官方模型。
- GPU 路径须显式 `--run-gpu --confirm-pilot-only`、实际剩余预算与 GPU 分配时间；clean checkout、单可见 A100、bf16 可用。包括分配后的安装/准备/空闲和清理预留，不猜用户余额；脚本内部墙钟检查 + 外层 watchdog，无无限重试。
- 手写小循环保留已审计 labels，不接会重建全序列 labels 的 Trainer/LM collator。batch3、两步 AdamW；HF loss 必须等于独立 shifted CE、监督非空、loss/grad 有限为正、**每次更新**都实际改变 LoRA 参数；只允许 LoRA trainable。
- checkpoint 带 Base 文件 fingerprint / 固定 revision / step=2，safetensors；保存后释放原模型，fresh Base + PEFT.from_pretrained 重载，逐 tensor 精确比对及指纹一致；固定 train prefix logits 在预设 bf16容差内一致，另记更新前后 delta。接口兼容当前研究模型 identity，但真实正式权重 GPU 加载仍待验。
- run 记录源码 hash、commit、计划/实际更新、loss/grad/tokens、checkpoint 身份、环境/设备、wall/分配成本、失败状态。Watchdog 强杀可能留下最后进度，外层 exit 与 GPU PID 查询为验收所必需，不能单看 result.json。
- 使用独立 Transformers/PEFT 栈，**不安装/启动 ART/vLLM**。结果始终 `ART_vLLM_verified=false`，成功状态也明确 ART/vLLM pending。正式配置/选点与共享 GRPO 起点没有实现或授权。

## 本地证据

`results/research/sft_entry_contracts_v1_20261012/` 留存 pytest、Ruff、fresh Pyright、冻结只读校验、旧 SFT 字节复核以及 preflight receipt/result。

- 专项 **61 passed**（原 SFT 测试29 + 新 pilot合同32）；全量 **567 passed / 11.21s**，使用与指南同版本的 pytest8.4.2 / pytest-asyncio1.2.0 / psutil7.0.0 / PyYAML6.0.2。
- 优化器合同用小型 CPU 张量，并用 **随机初始化的微型 Qwen2 + 真实 PEFT API** 验证 mask/loss/两次更新/save/fresh reload；无官方权重训练，不是模型能力或 A100 兼容结果。额外负控：篡改 mask、无监督、非有限/零 loss、无 trainable、墙钟超限、非 train、超预算与扩大配置。
- fresh 六文件 Pyright **0 errors / 0 warnings**；Ruff E/F/I/B（忽略 E501）通过。
- 活跃 LSP 起初重复报新模块 import missing；临时 extraPaths 没改善，已移除。只重启过期 Pyright语言服务后，新模块 definition 导航可正确解析，编辑后的五文件检查无 import finding；未用 import suppression/动态 import 逃避检查。此前的 false-positive 标记有历史隐藏提示，不将隐藏数当全清；独立 fresh Pyright与运行时导入均通过。
- preflight exit0，depths=[1,2,3]、不加载模型权重、不更新参数；Base 官方本地 snapshot 与全部模型字节留 hash。历史 preflight receipt 的 commit=09e2905 只表示基准，不冒称未提交源码已存在于该 commit；新版 result 另存实际 pilot 源码 hash。
- 原冻结 `--check` 通过，源锁/六份数据字节未变；旧 `data/research_sft_v1/` 字节 `--check` 仍通过，未覆盖旧产物。

模板 hash 有两种合同：SFT 数据 manifest 用 `digest(template)`（JSON编码字符串）=2e2d2512…；模型 identity/pilot 用原始 UTF-8 字节 SHA-256=cd8e9439…；同一模板、不同哈希编码，不能直接拿两种数值比较。消息/tool schema与R2保持一致。

## 推送边界

独立分支 `research/sft-pilot-v1-20261012`，不更改远程 master。由于此前研究版依赖尚未提交，本次需随入口一起提交研究源码/合同测试、版本配置、candidate/原字节 frozen 数据与协议文档；不重生成冻结源、不改变锚点。冻结 test 文件为原操作性公开 holdout 的原字节副本，存在于仓库不等于允许用它调试。

生成的12MB SFT token文件、官方权重、`.art`、旧反馈分析、大型历史 results均不推送；SFT预处理可在服务器重建。`.gitignore` 新增生成SFT目录/模型权重保护，不删除本地数据。新 CPU 合同的轻量 logs/receipt可随代码供复核；Colab safetensors 留Drive，反馈只复制白名单轻量证据到独立 feedback 分支，不自动合并。

远端是否成功以 git push 与 ls-remote 对同一实际 commit 的核对为准，不以本文件预先宣称成功。下一步由用户执行指南第1–4节，检查后第5节单独确认 pilot；失败保留日志、不自动重跑。ART/vLLM 真正加载测试另行授权。

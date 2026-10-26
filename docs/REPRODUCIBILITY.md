# 复现与发布范围

## 1. 本次发布包含什么？

- 最终HF训练/评测路径及其传递依赖、CPU回归测试。
- 冻结的train/validation/test、注册协议、四模型配置与55文件源码锁。
- [架构与训练方法](ARCHITECTURE.md)、[最终结果与局限](research_final_test_results_v1_zh.md)。
- `results/research/final_test_feedback_review_v1/`内的逐题CSV和必要的最终指标/案例/核验JSON。

## 冻结协议与当前状态

源码锁引用的少量诊断模块与三份历史协议保留原字节，不能为精简目录而破坏哈希核验：

- [初始预注册](research_preregistration_v1_zh.md)：记录早期五臂计划，不代表五臂实验已完成。
- [四模型test冻结附录](research_final_test_plan_v1_zh.md)：最终实际比较的事前身份、协议和统计规则。
- [原Colab执行协议](colab_research_final_test_v1_zh.md)：历史命令留档，不是要求再次运行test。

**以上文档中的“未执行”“未授权”是准备阶段状态，不是当前研究进度；四模型test已完成且已消费。** 当前执行结果以最终报告为准，不以历史预算当作现有GPU余额，不重新封存或改写旧锁。冻结数据中的`DATA_CARD.md`同样来自候选阶段，实际冻结身份以`freeze.json`及登记锁为准。

仓库历史压缩后，产物中的旧commit ID仅作原实验元数据，不能再作为可checkout入口；文件SHA-256和原源码锁保持不变。

## 2. 分层复现能力

| 层次 | clone后是否直接具备输入？ | 能证明什么？ |
|---|---|---|
| 逐题CSV统计复算 | 是；Python 3.10+标准库 | 发布表的分母、正确数、均值/SD、配对区间和调用增量 |
| 源码/数据锁核验 | 是；Python标准库 | 发布的冻结文件字节与原锁一致 |
| 数据、checker与训练合同CPU测试 | 是；需安装对应Python测试依赖 | 本地实现与回归合同，不等于新GPU训练验收 |
| 原始960条反馈重放/tokenizer核对 | 否；需原始反馈、原交付包及匹配Base缓存 | 保存行为、工具/checker和token统计复核，不是重新生成 |
| 模型加载/重新训练/硬件复现 | 否；需权重、指定GPU环境及历史产物 | 不由本次精简发布单独保证 |

### 无依赖统计复算

```bash
python3 scripts/summarize_final_test.py
```

预期：正确数144/181/171/170，三seed均值72.50%，样本SD约2.53448个百分点，平均增量12.50个百分点，配对95%区间[8.33,16.67]，工具调用增量约8.37438%。复算沿用CSV中SFT行保存的冻结任务顺序，以及固定seed20261011的1000次bootstrap；不要先按sample ID重排，有限次固定索引抽样的端点会随索引到场景的映射改变。

### 源码锁核验

在仓库根目录执行；只读文件，不解析test作模型评测：

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('.')
lock = json.loads((root / 'configs/research/final_test_v1.lock.json').read_text())
for name, expected in lock['files'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Hash mismatch: {name}')
print(f"PASS: {len(lock['files'])} frozen files")
PY
```

### CPU回归测试

最终实验使用独立HF环境，不需要ART/vLLM来运行以下研究测试。在隔离环境安装现有固定依赖（CUDA版torch需按机器选择合适安装方式，不要替换已经工作的GPU环境）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sft-pilot.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest \
  tests/test_final_test_summary.py \
  tests/test_research_dataset.py \
  tests/test_research_checker.py \
  tests/test_research_answer_grpo_v1.py \
  tests/test_research_grpo_native.py \
  tests/test_research_final_test_v1.py \
  tests/test_research_final_test_campaign_v1.py -q
```

这些测试使用合同/mock/小张量，不加载官方模型权重、不启动真实训练。部分旧测试依赖历史ART环境；不要将未运行的全量历史测试宣称为本次发布通过。缺少原始反馈时，依赖该反馈的历史回放用例可能跳过。

历史发布曾在独立导出目录的Python3.12 / torch2.8 / Transformers4.57.1环境中运行`tests/test_research_*.py`及`tests/test_final_test_summary.py`：**561 passed，1 skipped**，跳过项需要未入库SFT10反馈ZIP。该记录不是每次clone或整理后的新测试结果，也不包含模型重新推理或全量历史ART测试。

## 3. 训练与评测入口

| 入口 | 作用 |
|---|---|
| `scripts/train_research_sft.py` | 短程SFT，默认仅预检，显式GPU参数才更新 |
| `scripts/continue_research_sft.py` | 从已提交checkpoint继续到SFT30 |
| `scripts/train_research_answer_grpo_v1.py` | 三seed答案主导GRPO，默认预检；依赖原SFT产物 |
| `scripts/eval_research_baselines.py` | 开发集Base/SFT同协议评测 |
| `scripts/eval_research_final_test_v1.py` | 冻结四模型test历史执行器；不应再用于调参 |
| `scripts/review_research_final_test_feedback_v1.py` | 固定原ZIP的离线复核 |

入口是审计与复现源码，不是clone后即可自动启动训练的通用脚手架。固定模型身份、历史checkpoint元数据和显式确认参数刻意保留，不能为了让示例“跑通”而绕过它们。当前不提供已验证的GRPO优化器恢复CLI，也不宣称新硬件或任意checkpoint可无修改重放原训练。

## 4. 未入库的完整复核输入

原始审查器需要以下文件/目录：

1. `results/20261018T133835Z-final-test-feedback.zip`，SHA256为`cd434cd4e036d613710996817145708eeea7093e16e18ebe9fb9b73c8bef1868`。
2. 原交付包`results/research/final_test_preparation_v1/final_test_v1_bundle.zip`，用于核验原bundle manifest；当前脚本不能仅凭CSV替代。
3. 与远端manifest一致的完整Base缓存目录（含本地tokenizer资产及文件清单）；审查器只加载tokenizer，不运行模型forward。

上述产物**未随Git发布，也未提供公开下载链接**。有原产物的维护者将其放回指定位置后，可从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m scripts.review_research_final_test_feedback_v1 \
  --feedback results/20261018T133835Z-final-test-feedback.zip \
  --base-dir /absolute/path/to/matching-base-cache \
  --output results/local-final-review
```

输出目录必须不存在；不要覆盖既有审查证据。该过程复算已保存的test，不重新生成响应，也不授权训练或用test选模型。原始ZIP、原包与权重保持本地/Drive保管，未声称本轮已上传公开存储或验证了Drive第二份副本。

## 5. 如何理解已发布证据？

- `per_scenario.csv`：960行模型×场景派生统计，不是原始模型响应。
- `verification.json`：历史审查回执；其中“已重放960条”描述已执行审查，不表示CSV脚本重新执行了原始重放。
- `metrics.json`、`paired.json`、`strata.json`：主指标、配对比较和分层。
- `costs_and_robustness.json`：调用/tokens/耗时及探索性敏感性分析。
- `cases.json`：典型成功与回退的消息/trace；不等于960条全部原始输入输出。

最终结论及局限以[最终报告](research_final_test_results_v1_zh.md)为准；test已消费，新方法研究需要新的独立验证安排。

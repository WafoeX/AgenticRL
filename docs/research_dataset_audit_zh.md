# R1.1：旧模板可解性与数据容量审计

> 2026-10-11；旧实验源码 `09e2905f08e3e3d5a9dde8940ad782a4c08e4c35`。
> 本次只读审计现有工具/模板，并增加独立 CPU 审计脚本与测试。没有修改旧工具、模板、训练或数据，没有执行 GPU 推理。

## 1. 结论

**不能仅靠去重和换 seed 重建有规模的独立 benchmark。** 除数学和单步转换外，其余 10 类模板合计只有 **49 种任务文本**，且存在工具事实缺失、量纲错误、固定答案位置及合法捷径。

新研究采用用户已确认的路线：**每场景固定种子的合成环境快照 + 统一结构化 JSON 最终答案**。保留 5 类工具与旧实验默认行为，新研究环境独立版本化。详见 [数据与评测规格](research_data_eval_spec_zh.md)。

本审计不宣称已经修复旧模板，也不把参考工具可执行误当成最终答案正确。修复/替换在新研究版本实施，旧数据保留作为历史证据。

## 2. 可复核的旧数据事实

| 数据 | 行数 | 不同任务文本 | 预期调用深度 1 / 2 / 3 |
|---|---:|---:|---|
| train | 200 | 86 | 82 / 98 / 20 |
| validation | 50 | 37 | 23 / 22 / 5 |

- validation 中 **36/50 行**在 train 有精确相同任务；涉及 **24 种文本**。
- 没有 `data/scenarios_test.jsonl`。
- 文本去重仍不足以保证语义实例独立；共享 `__dynamic__` 的调用签名也不能直接当语义实例 ID。
- 输入文件 SHA-256 保存于 [机器可读审计结果](../results/research/legacy_audit/report.json)，旧数据哈希也记录于 [任务书](resume_research_taskbook_zh.md)。

## 3. 逐模板审计与新版本处理决定

容量是**旧生成器原始任务文本**的有限容量，不是语义等价去重后数量；例如加法交换操作数会进一步减少独立实例。非数学域从模板的随机分支完全枚举，数学容量通过受保护的参数域公式计算：`3 × 990 × 98 = 291060`。

| 旧模板 | 深度 | 原始容量 | 审计结论 | 新研究版本的处理 |
|---|---:|---:|---|---|
| `age_calculation` | 2 | 4 | Guido 查询没有其出生年；使用隐藏 `_birth` 标签。Einstein 在 2020/2024 的实际年龄表述另有逝世日期歧义；仅有年份也不适合精确周岁 | 改为合成人物“目标年份减出生年份”，任务明确计算定义；search 返回该人物出生年，calculator 消费结果 |
| `weather_convert` | 2 | 5 | 天气与转换可执行；但 `get_weather(units=fahrenheit)` 一次即可合法回答，不能把该答案记错 | 快照温度可变；保留合法一调用路径。答案正确率与指定两步链路分开；另有必要依赖任务用于机制分析 |
| `population_math` | 2 | 5 | 人口/面积均有有效记录，分母为正；具备 lookup→calculator 动态路径，但实例极少且事实可记忆 | 改为合成记录，任务明确 people/km²；使用真实 observation 的两操作数 |
| `simple_math` | 1 | 291060 | `+/-/*` 均可执行；容量充分，但当前数据记录不足以审计参数和交换等价分组 | 保留原子能力，结构化参数与等价签名；区分同分布与参数范围 holdout |
| `simple_weather` | 1 | 7 | 工具有效，但“weather like”需要哪些字段没有固定答案口径 | 明确查询温度/天气条件等指定字段，不用模糊综合描述做自动准确率 |
| `simple_search` | 1 | 5 | 当前五个查询可命中相应知识；答案类型混杂，工具同时含不同单位/近似值 | 合成记录查询指定字段，固定权威字段和精度，不把近似别名混作 gold |
| `search_and_convert` | 2 | 1 | 第二步 value 固定 `149600000`，没有 observation 消费检查；检索同时给约 9300 万英里，存在直接回答路径 | 合成搜索数值，以实际返回值构造转换；可允许查询直接目标单位时的替代答案，严格协议另计 |
| `country_density_convert` | 3 | 4 | 严重量纲错误：把 people/km² 当 km 执行 km→miles；工具成功不是任务正确 | 使用显式密度单位 people/km²→people/mile²，乘 `(1.609344)^2`；在新版本扩展 convert_units 支持，保留工具类别 |
| `weather_compare` | 2 | 9 | **9/9 答案都是 slot B**，无需查询即可猜中；两次查询独立，不是调用参数动态依赖 | 随机化温度与实体顺序，平衡 A/B，单列相等情况；并行/交换独立查询作为语义合法解 |
| `simple_convert` | 1 | 100 | kg→lbs 可执行，整数取值少、方向单一 | 扩展可解取值与现有转换方向；单位与精度显式记录 |
| `simple_lookup` | 1 | 5 | GDP 查询可执行，但任务未明确币种/数量级，库是 billion USD | 改为明确字段与 billion USD；合成实体，避免真实 GDP 事实混淆 |
| `country_compare` | 2 | 4 | **4/4 答案都是 slot B**；同样是独立查询 | 可变人口、随机实体顺序与 A/B 平衡；答案、严格顺序、依赖分别统计 |

共 12 类均已审计。新版本可以替换旧任务表达/参数空间，不能声称仍是同一旧 benchmark 的直接成绩对比。

## 4. 已执行的语义探针

### 4.1 出生年并不来自检索

对所有四个年龄实例执行参考工具调用：

- Einstein 查询返回包含 1879 出生年的记录。
- `Guido van Rossum birth year` 与 Einstein 检索键共享 `birth` / `year`；当前按 token overlap 取最大值，结果返回 **Einstein** 的出生年。
- 参考 calculator 却使用 `_birth` 中的 **1956**，不是工具返回事实。

这不等同于“现实世界不存在该事实”，而是当前可控工具环境没有提供该事实。新版本不得用隐藏答案补齐模型根本拿不到的 observation。

### 4.2 平方单位错误

若密度为 `d` people/km²，正确换算为 `d × (1.609344)^2` people/mile²。

旧链路执行的是 `d × 0.62137`。四个国家都能顺利执行但相对正确答案误差约 **76%**。参考链路成功与工具无报错完全不足以证明语义正确。

### 4.3 固定第二选项

枚举天气比较 3×3 与国家比较 2×2 的全部实例：两类均只有 slot B 获胜，没有 slot A 或 tie。新版本必须对标签位置和比较值做审计；不能仅随机换同样偏置的文字。

### 4.4 合法替代路径

对 weather_convert 的全部五个城市，直接查询 Fahrenheit 与“查 Celsius 再转换”结果相同。新评测应记录：`answer_correct=true`，但若参考两步协议未满足，`strict_chain_complete=false`；不是答案假阳性。

### 4.5 动态判定的覆盖不足

旧 `src/reward_v2.py::_dynamic_matches` 主要支持 weather→数值、calculator/convert→数值、lookup→人口/面积除法；不支持任意 search 返回事实消费或新组合表达式。

因此新环境引入后必须单独版本化并补齐研究用 V2 的依赖规则与测试，再冻结奖励。不能因为同一旧函数对错误密度链路给分就认为它是独立答案 oracle；也不能悄悄修改 V2 后仍宣称是未变的旧奖励。

## 5. 审计产物与复现

- `src/dataset_audit.py`：不导入 ART/reward，检查数据结构、文本重叠、原始容量，执行语义探针。
- `scripts/audit_legacy_dataset.py`：默认只打印摘要；可将 JSON 写到独立审计目录。拒绝向旧数据路径写报告，已有报告需要显式覆盖。
- `tests/test_dataset_audit.py`：21 项 CPU 测试，包含容量/模板漂移、异常 JSON、动态占位符、输入不变、位置捷径、量纲错误和 CLI 输出路径保护。
- `results/research/legacy_audit/report.json`：原始探针、各模板计数与输入哈希，不含模型能力评测。

在仓库根目录：

```bash
python scripts/audit_legacy_dataset.py
# 初次写出报告；已有报告需要另加 --overwrite，不能覆盖旧数据。
python scripts/audit_legacy_dataset.py \
  --output results/research/legacy_audit/report.json
python -m pytest tests/test_dataset_audit.py -q
```

验证结果：

- 新增审计测试：**21 passed**。
- 全量 CPU 回归：**194 passed in 1.55s**。
- 新增 3 个 Python 文件：独立 fresh Pyright **0 errors / 0 warnings**；Ruff `E,F,I,B`（忽略 E501）通过。
- 本机原 Python 3.14 环境缺 psutil，最初全量测试 collection 失败；没有升级 GPU 栈，改用隔离 `uv run --no-project --with pytest --with pytest-asyncio --with psutil` 后通过。新增审计本身只需要标准库。
- 会话语言服务对新增模块曾提示 MissingImports；真实脚本执行、21 项直接导入测试及独立 fresh Pyright 均通过，属于新增文件解析缓存误报，不添加源码忽略注释。

**验收边界：**R1.1 的审计与新版本修复方案已完成；R1 数据生成、实际语义修复与冻结尚未完成。没有训练增益、模型推理或 GPU 成本测量结论。

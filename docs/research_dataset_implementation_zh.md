# Research V1：schema、确定性工具环境与生成器验收

> 2026-10-11；完成任务书 R1.2–R1.5，以及候选数据的 R1.7 留档。R0.4 的目标规模/容量已落实。**不是 R1 全部完成或独立模型评测完成**：R1.6 干预诊断、R1.8 test 冻结、R2 最终答案 checker 尚待实施。
> 不修改旧工具、旧场景或旧 train/val；无 GPU、SFT/GRPO、模型 test 推理；不自动 commit/push。

## 1. 实现与路径

| 路径 | 功能 |
|---|---|
| `src/research_schema.py` | 字段/类型、有限值、身份/hash、答案类型/单位/容差、参考执行和 DAG 校验；模型消息白名单 |
| `configs/research/schema_v1.json` | Draft 2020-12 JSON Schema；包含快照、调用、轨迹、依赖图、答案结构 |
| `src/research_environment.py` | 五类独立工具及 schema；不可变序列化快照、局部状态、严格参数校验、安全 AST 算术 |
| `src/research_generator.py` | 12 个修复后基础 recipe + 4 个保留组合；有限实例池、先分组分配、表达渲染、执行参考、独立 Decimal 公式 |
| `configs/research/dataset_v1.json` | seed、域、配额、recipe/工具路径、三套表达注册表 |
| `src/research_audit.py` | 重新推导每条记录而非信任缓存签名；容量/配额/执行/泄漏/holdout/比较平衡、实体与训练事实暴露诊断 |
| `scripts/generate_research_dataset.py` | CPU 构建与只读 `--check`；原子写入、禁止覆盖、限制输出位置、代码/数据/旧文件 hash 留档 |
| `tests/test_research_dataset.py` | 81 项新增 CPU 测试 |
| `data/research_v1_candidate/` | 三份候选 JSONL、`manifest.json`、`audit.json`、`DATA_CARD.md`；**test 未冻结** |

运行时只需 Python 标准库。没有导入 ART/torch/vLLM 或联网工具；不改变旧 rollout、reward 或训练入口。新数据不能直接当旧 `id/task/expected_calls` 数据接口无审查地送进旧训练。

## 2. 记录、分组与 prompt 边界

每条记录包含任务/表达/recipe 身份、输入参数、split/泛化类别、固定快照及版本/hash、语义/去实体名值/轨迹签名、依赖图、真实 observation 参考轨迹、答案规格与公式 gold。

- `sample_id = rv1-<semantic_signature>`，与整组语义身份绑定，而不是任意编号。
- 语义身份包含操作 recipe、任务输入及**需要的环境事实**，不使用无关随机天气字段制造“独立实例”。
- 去实体名值签名检测仅换名字的重复；算术 `+/*` 交换等价规范化，`-/÷` 保留顺序；比较按实体/事实配对规范化，反转 A/B 不产生新组。
- 完整轨迹签名使用具体参数、算术 AST 规范化及去实体名的语义信息；独立调用顺序规范化，不把共同的单个前缀调用当完整任务重复。
- 实体来自共享 2048 名匿名池：固定 SHA-256 映射隐藏生成索引，名称不直接编码温度/人口等参数。模型看不到 seed/index。
- `model_messages()` 只发送固定协议和自然语言 `task`；不序列化快照、参数元数据、gold、参考轨迹或预期调用。快照仅供本地工具执行器使用。
- 一个参考调用占一个 turn；后续动态参数必须由刚执行的 observation 构造。测试直接修改生产事实，证明参考构建不使用旧 hidden gold 填参数；这不是已实现的模型因果干预评测。

## 3. 工具语义与可解性

- snapshot 复制/序列化后不可变，返回的字典是防御性副本；100 次跨 8 线程交替查询两环境无串扰。
- `search` 精确契约：`<entity> birth_year` / `<entity> distance_km`，返回实体、字段、值与单位；不沿用旧模糊词重叠搜索。
- 年份差明确“目标年 − birth_year”，不声称精确周岁。
- 密度明确人口/正面积；平方密度转换使用 `1.609344²`，不是长度转换。
- 单位固定常量：mile=1.609344 km、lb=0.45359237 kg、foot=0.3048 m；温度公式固定，转换舍入六位小数。
- calculator 仅允许有限数字和 `+-*/` 的有界 AST；禁止 bool、NaN/Infinity、名字、函数调用、幂、除零和过长表达式。
- 工具 schema 拒绝多余字段/缺字段/非法枚举；非法快照（零面积、错误单位、布尔人口等）构造即失败。
- 所有 16 recipe 在三种参数域及边界索引都真实执行测试；比较平局 fixtures 返回 `tie`。
- internal oracle 使用独立 Decimal 公式；工具执行与公式不共用转换实现，配合手算温度、长度、质量、密度样例。共享原始事实输入的共同错误风险仍存在。

**R2 尚未完成**：`answers_agree()` 只校验内部参考/公式，未实现最终模型回复的严格 JSON 解析、替代合法解、冲突、多答案或 adversarial checker。

## 4. 参数域、配额和容量

`dataset_v1.json` 是公开候选注册表，默认 seed=20261011，每 recipe/domain 1024 个有限 indexed candidates。

基域/validation OOD/test OOD 的具体值由注册基数与代码中固定偏移定义。例如：

| 数值 | base | val OOD | test OOD |
|---|---|---|---|
| 温度（包含比较的较高值） | −10…15.23 | 25…50.23 | 60…85.23 |
| 出生年 | 1900…1979 | 1800…1879 | 1700…1779 |
| 目标年 | 2020…2032 | 2100…2112 | 2200…2212 |
| distance_km | 100…202.3 | 1000…1102.3 | 10000…10102.3 |
| 单转换输入 | 1…1024 | 5001…6024 | 10001…11024 |
| area_km2 | 100…196 | 2000…2096 | 5000…5096 |

人口、GDP、算术操作数、倍率和偏移也有分别注册且互斥的数值段；配置校验拒绝域重叠。天气等均是**虚构实验事实**，不是现实城市数据。

同深度可用 family 按名称排序做商/余数配额，精确 family/bucket 计数写入 `audit.json → allocation.quotas`。没有通过丢样本缩小分母。

| split | 总量 | 1-call | 2-call | 3-call | 动态参考依赖 |
|---|---:|---:|---:|---:|---:|
| train | 1200 | 360 | 480 | 360 | 680 |
| validation | 180 | 54 | 72 | 54 | 106 |
| test candidate | 240 | 72 | 96 | 72 | 144 |

validation 的 iid/parameter/wording/composition = 90/30/30/30；test 四类各60。各分层深度与已确认规格完全一致。

枚举 **40 个有需求的 recipe/domain 池**，每池去实体名后仍有1024个独立值/操作组。最紧池为基础密度三步，需求411，剩余下界613；分组/文本冲突后仍完整填满全部配额。分配候选被拒绝5190次（已分组）+17次（规范化文本冲突），这是候选分配重试，不是已输出任务丢弃。无法填满将报错，不写半份数据。

比较在每 split/分层/recipe 的非平局 A/B 数差≤1；主候选集不含平局，平局由 fixtures 覆盖。

## 5. 泄漏与 holdout 验收

train/validation、train/test、validation/test 三对中，下列交集均为0，split内部也无重复：

1. 精确任务文本；
2. NFKC/大小写/标点/空白规范化任务文本；
3. 语义实例组；
4. 去实体名的值与操作签名；
5. 规范化完整参考轨迹签名。

审计重新从配置/candidate index 构建快照、表达、调用、图、gold 与签名，拒绝仅修改标签后重新计算 hash 的伪记录；也拒绝缺样本、未知字段/recipe、reserved recipe 放错桶、失衡比较、非有限值、未执行/伪造 observation。

- validation 保留 GDP×倍率、天气+偏移→F；test 保留测量×倍率、距离+偏移→miles，recipe 不进入 train，test recipe 不进入 validation。
- 两步组合沿用训练见过的工具序列；三步组合序列在训练未出现。全部原子工具、`+/*` 运算与所需转换方向在 train 覆盖。
- wording holdout 使用独立语法表达模式/ID，同时重新分配独立语义实例，不复制 train 实例只换措辞。
- parameter holdout 的实体池与其他分层共享，但不保证所有实体已在 train 出现。test parameter 中57个实体有29个未在train出现，val parameter 中29个实体有16个未出现：**不能宣称纯数值单因素泛化**。

### 共享环境事实不是“全部内容未见”

在真实训练参考 observation 暴露过的事实中，validation/test 的 parameter 层所需事实交集为0；IID、表达与组合层存在部分共享事实。例如 test IID 的50个需查事实任务中13个至少一项曾在train返回，10个所需事实全部曾返回；test组合60个中13个事实曾返回。

这不等于禁止的等价任务/答案监督实例重叠，却提供环境记忆捷径。`audit.json → train_observation_fact_exposure` 和数据卡完整单列，不宣称所有事实未见或无任何形式的信息复用。后续冻结时必须保留这一边界。

此外，基础3-call全为密度转换，存在深度/family混杂；weather_convert允许一调用Fahrenheit替代答案；每family小分层样本很少，不能将总体深度曲线泛化为通用推理能力。

## 6. 验证证据与重现命令

最终本地验收：

- 全量 **275 passed in 5.98s**，其中81项本轮研究测试；旧194项无回归。
- 六个新增Python文件 fresh Pyright：0 errors / 0 warnings；Ruff `E,F,I,B --ignore E501` 通过。
- Draft 2020-12 schema 本身及全部1620条记录通过独立 `jsonschema` 验证。
- 构建和 `--check` 均退出0；后者重新生成并逐字节比较包括manifest的六个产物，不只是核对已保存hash。
- 编辑器新模块路径缓存告警经真实import、fresh Pyright核实并登记误报；`.execute()`的SQL告警误认本地工具分派器，不存在SQL执行。未添加忽略注释。
- 旧两份数据hash与任务书第2节完全一致，旧 `tools.py/scenarios.py` 前后hash一致。

从仓库根目录运行（已有输出时只读检查；脚本禁止覆盖）：

```bash
python3 scripts/generate_research_dataset.py --check
uv run --no-project --with pytest --with pytest-asyncio --with psutil python -m pytest -q
```

新机器尚无候选目录时：

```bash
python3 scripts/generate_research_dataset.py
```

需要额外副本/不同seed验证时，先保留原配置，使用显式新配置和专用结果目录；不得原地覆盖候选数据或旧数据。生成器不会自动冻结test。

manifest 基准代码commit为 `09e2905f08e3e3d5a9dde8940ad782a4c08e4c35`，但本轮实现尚未提交，**commit本身不包含新实现**。权威工作树源文件联合hash为：

`1067d65393328bff066aa39cd928367432cba2df96cff5d5f73e1d66c8415383`

候选JSONL SHA-256：

- train：`b98fc61cfae88426e065ff817db9757650944c76f3268298de82eb76a097048e`
- validation：`941132e1c93ce26cba6417eac7892c11b7c64c2fc14d3c21b02902e1dff70261`
- test：`08040a4e2e7b7c410f17937496a22bccd09f925e4c5dba35da2f027380410a52`

## 7. 下一关卡

实施 R1.6 同组配对 observation 扰动/环境变体开发诊断，然后 R2.1–R2.4 独立模型最终答案 checker、语义合法替代与对抗轨迹。候选数据通过后仍不越过 oracle/protocol/test冻结关卡，不启动SFT/GRPO或消费 GPU。

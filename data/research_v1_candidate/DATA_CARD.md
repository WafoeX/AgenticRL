# Research V1 候选数据 CPU 验收

**状态：构建/容量/泄漏/参考执行通过；test 尚未冻结，无 GPU 或模型评测。**

## 数据计数

| split | 总数 | 1-call | 2-call | 3-call | 动态依赖 |
|---|---:|---:|---:|---:|---:|
| train | 1200 | 360 | 480 | 360 | 680 |
| validation | 180 | 54 | 72 | 54 | 106 |
| test | 240 | 72 | 96 | 72 | 144 |

## 泛化分层

| split | 分层 | 数量 |
|---|---|---:|
| train | train | 1200 |
| validation | composition | 30 |
| validation | iid | 90 |
| validation | parameter | 30 |
| validation | wording | 30 |
| test | composition | 60 |
| test | iid | 60 |
| test | parameter | 60 |
| test | wording | 60 |

## 容量与泄漏

逐个枚举 40 个 recipe/domain 池，每池最多1024候选；按去实体名的值/操作签名计数。
最紧需求池剩余下界：613。这是注册有限池容量，不是所有理论世界的容量。
全局先分组再分配，不删除已输出样本；候选冲突只计分配重试，无法填满配额直接失败。
分配候选冲突计数：`{"already_allocated_group":5190,"normalized_text_collision":17}`。

跨 split 精确文本、规范化文本、语义组、去实体名值签名、规范化完整轨迹签名重叠全部为0；split内部同样无重复。
全部 1620 条参考已重放并与独立 Decimal 公式核对。JSON审计含每类配额、比较A/B平衡和组合序列是否在train出现。

## 边界

- 生成的oracle仅用于参考构建验收；尚不是R2的最终模型答案解析/判定器。
- 不包含R1.6干预诊断、不冻结test、不接入旧rollout/SFT/reward；旧数据及工具hash在manifest中留档。
- 实体来自共享命名池，但部分holdout实体可能未在train出现；audit.json单列实体新颖性，不能声称纯数值单因素OOD。
- 两比较recipe在每个分层中A/B平衡；正式候选集无tie，CPU fixtures测试tie。
- weather_convert的一调用Fahrenheit替代解合法；参考深度不等于必要调用次数。
- 三步基础任务只有密度转换，家族/深度存在混杂；不能把全局深度曲线当通用多步能力。
- 合成快照共享事实输入，不是外部盲测；checker/reward/protocol未锁定前不得宣称冻结测试。

## 训练 observation 的事实暴露诊断

允许共享环境事实不等于等价任务标签泄漏，但可能存在记忆捷径；只统计真实train工具返回过的字段，不统计未查询的隐藏快照。
匿名实体来自与参数索引无直接可见关系的固定hash映射；模型prompt不包含生成seed/index。

| split | 分层 | 需隐藏事实的任务 | 至少一项train见过 | 全部train见过 |
|---|---|---:|---:|---:|
| validation | iid | 75 | 18 | 13 |
| validation | parameter | 26 | 0 | 0 |
| validation | wording | 26 | 8 | 5 |
| validation | composition | 30 | 8 | 8 |
| test | iid | 50 | 13 | 10 |
| test | parameter | 50 | 0 | 0 |
| test | wording | 50 | 12 | 9 |
| test | composition | 60 | 13 | 13 |

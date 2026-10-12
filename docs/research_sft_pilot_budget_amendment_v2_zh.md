# SFT 开发 pilot 启动检查修订 v2

2026-10-12，用户明确要求取消手填 GPU 剩余小时/分配 UTC 时间，优先让程序启动。本修订仅针对两次更新的开发 pilot，不改变已冻结数据、原预注册文件、正式训练/最终 test 就绪状态。

- 指南第2格删除两项 input 与对应时间/余额断言，自动记录本格开始时间。
- CLI 不再调用 check_budget，也不要求 --remaining-gpu-hours / --gpu-allocated-at。旧参数仅兼容接收并忽略，错误/未来日期不会阻断运行或在 finally 中再次解析失败。
- 仍需 --run-gpu --confirm-pilot-only、clean checkout、单A100、固定3条train/两次更新、600秒上限、独立新输出目录及原loss/身份/reload校验；不授权正式SFT/GRPO，不自动重试。
- result 自动记录 started_at / ended_at / wall_seconds / pilot_process_wall_hours。实际GPU分配时间、此前消耗与总剩余额度为未知，total_GPU_budget_verified=false；不会用当前时间冒充分配时间，也不能声称总20小时成本已核验。程序退出后仍需手动关闭计费runtime。

新增3个入口回归：不提供预算参数、传未来日期、传0余额/非法日期时，均越过已取消的手填检查进入preflight；用替身截断在模型加载之前，不消耗GPU。全量570passed10.75s；Ruff及fresh两文件Pyright通过。无本轮模型/GPU训练或推理，A100运行仍待用户反馈。

原入口报告和任务书之前的预算门禁描述为v1历史事实，本v2在开发pilot启动行为上取代它；独立的 check_budget 工具函数及其历史单元测试仍保留，但GPU入口不使用它。正式实验预算附录不在本修订范围。

已有服务器checkout更新同一 `research/sft-pilot-v1-20261012` 分支即可，无需删除环境、重新克隆或覆盖旧run。使用更新指南的第2格与第5格；已安装依赖不必再次创建venv。不要在旧第2格继续输入日期，也不要混用旧GPU命令的预算环境变量。

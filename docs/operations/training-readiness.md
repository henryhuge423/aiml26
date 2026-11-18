# 训练准备与完成状态

2026 年 11 月 15 日更新 · [English](training-readiness.en.md)

`expanded_20261113` 已完成数据核验、GPU 恢复预检、四组训练及开发集评价。

`followup_20261115` 使用同一冻结清单，六方案、三种子的 18 组训练和开发评价已全部完成，每组 2,400 次参数更新。混合退化、65% 原来源采样和仅失焦三项 CUDA 连续／恢复预检均通过，权重最大绝对差为零，逐步记录和状态一致。完成证据见[执行状态](../../runs/followup_20261115/pipeline/status.json)、[预检记录](../../runs/followup_20261115/preflight/verification.json)和[完整结果分析](../progress/2026-11-15/completed-experiments/completed-results.zh.md)。

| 核查项目 | 当前证据 |
|---|---|
| 数据版本 | 24,491 张疾病图像，清单哈希以 [cohort_verification.json](../../data/prepared/expanded_20261113/cohort_verification.json) 为准。 |
| 划分 | 15,480 训练、2,046 模型验证、2,053 规则开发、3,470 内部及新增保留测试、705 ACRIMA 和 737 HYGD。 |
| 图像关联 | 已知患者与确认的关联组在同一用途；文件哈希及 RGB 核验覆盖全部图像。 |
| 退化方法 | 照明、失焦、散射实现及真实配对校准见[退化审计](../progress/2026-11-13/training-results/degradation-audit.zh.md)。 |
| GPU 恢复 | 相同六次更新的连续与中断恢复检查，权重最大绝对差为 0，损失序列一致。 |
| 训练完成 | 11 月 13 日四组各 1,200 次更新；11 月 15 日 18 组各 2,400 次更新，全部完成开发评价。 |
| 评价 | 共同原图开发集与固定合成压力子集，选中权重和预算结束权重分别保存结果。 |

运行命令见[运行手册](training-runbook.md)，结果解读见[训练报告](../progress/2026-11-13/training-results/training-results.zh.md)。新的配置或数据版本需与自身输入对应的核验记录。原 `formal_v2` 的准备说明保存在[历史记录](../historical/2026-11-09/training-readiness.md)。

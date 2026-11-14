# 扩展数据训练记录

[English](README.en.md) | 中文

四组各完成 1,200 次优化更新，每 300 次在共同模型验证集上评价。`best.pt` 按 11 个来源的平均 AUROC 选择，`last.pt` 为第 1,200 次更新权重。扩展无退化组选中第 300 次更新，其余组选中第 1,200 次更新。原图与固定合成压力图的开发结果分别保存。

[进度总览](../../docs/progress/2026-11-13/progress-overview.zh.md) · [训练报告](../../docs/progress/2026-11-13/training-results/training-results.zh.md) · [统计分析](../../docs/progress/2026-11-13/experimental-analysis/experimental-analysis.zh.md)

`comparison.json` 保存选中权重的评价，`budget_end_comparison.json` 保存相同第 1,200 次更新的评价。各组 `run_summary.json` 记录更新数与采样次数，`development_metrics.json`／`last_development_metrics.json` 分别记录这两种权重的开发指标。

[四组配置](../../configs/expanded_20261113/) · [GPU 恢复预检](preflight/verification.json) · [运行手册](../../docs/operations/training-runbook.md)

权重、逐图预测和完整日志保留本地；上述汇总结果随仓库保存。下一轮使用新的配置与输出目录。

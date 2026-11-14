# 2026 年 11 月 13 日：新数据集与退化实验

[English](README.en.md) | 中文

今天的工作由数据准备、退化方法核验、四组训练和结果分析组成。先读总览，再沿以下顺序查看证据与方法。

| 阅读顺序 | Markdown | PDF | 内容 |
|---|---|---|---|
| 总览 | [进度总览](progress-overview.zh.md) | [PDF](progress-overview.zh.pdf) | 当前完成状态、主要结果、数据规模和下一轮重点。 |
| 1. 数据与方法 | [数据及退化审查](data-review/data-and-degradation-review.zh.md) | [PDF](data-review/data-and-degradation-review.zh.pdf) | 45 个公开来源、9,070 张新增图像、标签与图像关联、退化研究和评估设计。 |
| 2. 训练与评价 | [训练结果](training-results/training-results.zh.md) | [PDF](training-results/training-results.zh.pdf) | 联合训练清单、四组 1,200 次更新、原图和质量评价；附[数据审计](training-results/data-audit.zh.md)及[退化审计](training-results/degradation-audit.zh.md)。 |
| 3. 结果解释 | [统计分析](experimental-analysis/experimental-analysis.zh.md) | [PDF](experimental-analysis/experimental-analysis.zh.pdf) | 来源配对差异、采样次数、选模影响、失焦结果及真实质量病例。 |

三份详细报告记录同一天不同阶段的工作：新增池审查完成后，图像才经分组分配到训练、验证、开发及测试用途。总览和统计分析的主要比较均使用第 1,200 次更新权重；训练报告同时记录各组在模型验证集上选中的权重。

每份主报告的中英文 `.md`、`.pdf` 和 `.tex` 同名保存，图表、汇总数据及核验记录放在对应主题目录。[统计 JSON](experimental-analysis/analysis.json)和[来源差异 CSV](experimental-analysis/paired-source-contrasts.csv)支持表格中的数值核查。

[全部进度报告](../README.md) · [文档导航](../../README.md) · [下一步工作](../../operations/next-steps.md)

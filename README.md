# Eye Sentinel

[English](README.en.md) | 中文

Eye Sentinel 研究如何结合眼底照片的青光眼分类分数和图像质量分数，选择需要人工复核的照片。项目将比较质量引导复核与另一疾病模型按不确定性选择相同数量照片的方案，评价复核后自动结果的错误率与青光眼漏诊。

## 当前进度：2026 年 11 月 13 日

本轮完成公开数据审查、关联分组、训练划分及四组 EfficientNet-B0 实验。新增池含 9,070 张图像；联合清单含 15,480 张训练图像、2,046 张模型验证图像和 2,053 张规则开发图像。四组均完成 1,200 次优化更新，比较原数据、扩展数据及 30%／50% 混合退化。

在同一规则开发集、第 1,200 次更新权重下，扩展数据使 KEH 的 [AUROC](docs/progress/2026-11-13/progress-overview.zh.md#已观察到的结果) 从 0.6627 提高到 0.9333，LAG 从 0.9465 提高到 0.9890；AIROGS 和 ODIR 的表现下降。50% 退化改善合成失焦的排序并减少误报，同时增加阈值 0.5 下的漏诊。12 张真实低质量开发正例的检出由 7/12 提高到 12/12，三个扩展组的检出数一致。

下一轮重点为原来源采样量、模型选择规则和真实低质量阴性病例；分类阈值、人工复核规则与最终测试尚待完成。此前两套疾病模型、原质量模型及 EyeQ 质量模型均完成 20 轮，见 11 月 12 日历史报告。

完整结果、样本背景和下一步安排见[中英文进度总览](docs/progress/2026-11-13/README.md)（[中文 PDF](docs/progress/2026-11-13/progress-overview.zh.pdf) · [English PDF](docs/progress/2026-11-13/progress-overview.en.pdf)）。当天的数据审查、训练报告与统计分析已集中在同一日期目录。

## 文档与运行入口

- [文档导航](docs/README.md)：当前操作文档、研究材料和历史记录。
- [进度报告索引](docs/progress/README.md)：按日期阅读中英文报告、PDF 和图表。
- [当前运行手册](docs/operations/training-runbook.md)与[下一步清单](docs/operations/next-steps.md)。
- [项目提案](proposal-1106.pdf)与[完整目录说明](docs/DIRECTORY.md)。

## 数据与代码

原疾病数据来自 AIROGS-Light、ODIR-5K 和 Kaggle Eye Diseases。新增池保留 LAG、GRAPE、KEH、BEH、PAPILA、FIVES、MSHF、Fundus-AVSeg、LES-AV、HRF、CRFO、Chákṣu 和 3 张 AIROGS 原始转诊阳性图的标签及分组。质量模型使用 FIQS 和 EyeQ。来源标签定义、视野类型和确认的图像关联在数据清单中分别记录。

`configs/` 和 `training/` 存放配置与训练实现，`tools/` 和 `tests/` 存放处理工具及自动化检查。`data/` 保存数据清单与审计，`models/` 和 `runs/` 保存本地权重与运行结果。仓库同步代码、文档、图表及选定的结果汇总；原始数据、权重和完整日志保留在本地。

[数据工作区说明](data/README.md) · [联合数据版本](data/prepared/expanded_20261113/README.md) · [新增数据池](data/prepared/expansion_20261113/README.md)

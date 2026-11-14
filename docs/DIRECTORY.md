# 项目目录说明

[English](DIRECTORY.en.md) | 中文

Eye Sentinel 仓库位于课程工作区的 `team_project/`。项目文档按进度、当前操作、研究及历史记录组织；数据和训练程序使用的路径保持原位置。

| 位置 | 内容与入口 |
|---|---|
| `README.md`、`README.en.md` | 项目目标、最新结果与主要入口。 |
| `proposal-1106.pdf` | 当前项目提案。 |
| `docs/README.md`、`docs/README.en.md` | [文档导航](README.md)。 |
| `docs/progress/YYYY-MM-DD/` | 按日期保存的中英文进度报告、PDF、LaTeX、图表与汇总数据。 |
| `docs/progress/2026-11-13/` | [新数据集与退化实验整体材料](progress/2026-11-13/README.md)，含 `data-review/`、`training-results/`、`experimental-analysis/`。 |
| `docs/operations/` | 当前训练运行手册、准备核验、方法协议及下一步工作，中英文对应。 |
| `docs/research/` | 数据扩展研究与结构化调研提纲。 |
| `docs/historical/` | [按日期归档的审计和旧操作记录](historical/README.md)，以及早期数据介绍与讲稿。`planning/` 保留原设计和执行计划。 |
| `configs/` | 训练配置；`formal_v3/` 是此前训练，`expanded_20261113/` 是本轮四组实验。 |
| `training/` | 数据加载、模型、训练、质量监督和眼底退化实现。 |
| `tools/`、`tests/` | 数据核验、训练、报告工具及自动化检查。 |
| `data/` | [数据说明](../data/README.md)、来源文件、清单与审计。 |
| `models/` | [模型资源说明](../models/README.md)，本地初始化、教师和训练权重。 |
| `runs/` | 各轮训练状态、指标、预测与权重；精选汇总进入 Git。 |
| `notes/` | 前期研究、可行性和课程评审笔记。 |
| `archive/`、`references/` | 早期提案与本地参考资料。 |
| `.envs/` | 本地运行环境。 |
| `tmp/`、`.cache/`、`.build/`、`.test_tmp/` | 临时文件、缓存、PDF渲染与检查产物。 |

文档主题目录使用小写英文和连字符，进度目录统一使用日期。历史正文保留，移动时更新引用。原始数据、权重、完整运行日志、环境及临时产物留在本地；GitHub 同步代码、配置、文档、PDF、图表和选定的汇总结果。

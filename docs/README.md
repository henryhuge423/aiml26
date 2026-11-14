# 文档导航

[English](README.en.md) | 中文

先阅读[最新进度总览](progress/2026-11-13/progress-overview.zh.md)，再按需要查看数据审查、训练结果与统计分析。操作文档描述当前入口，历史记录保留各阶段的版本与实验背景。

| 分类 | 入口 | 内容 |
|---|---|---|
| 进度报告 | [按日期索引](progress/README.md) | 中英文 Markdown、PDF、LaTeX、图表和汇总数据。 |
| 当前操作 | [运行手册](operations/training-runbook.md) | 本轮数据版本、配置、状态核查及报告生成。 |
| 后续工作 | [下一步](operations/next-steps.md) | 采样与退化对照、模型选择、复核规则和最终评价。 |
| 准备核验 | [训练准备](operations/training-readiness.md) | 当前实验输入及完成状态的检查依据。 |
| 方法协议 | [研究训练与评价协议](operations/training-protocol-2026-11-09.md) | 分数定义、数据用途、人工复核比较及评价指标。 |
| 数据研究 | [扩展数据研究](research/data-expansion-2026-11-11.md) | 11 月 11 日的来源研究，及 [outline](research/data-expansion/outline.yaml)、[字段定义](research/data-expansion/fields.yaml)。 |
| 历史记录 | [历史索引](historical/README.md) | 早期提案、审计、训练快照及原操作文档。 |
| 全仓库目录 | [目录说明](DIRECTORY.md) | 代码、数据、运行结果和项目背景材料的位置。 |

`progress/` 下的目录统一使用 `YYYY-MM-DD`。同一天的相关工作放在同一目录，按主题使用小写英文和连字符命名；主报告以 `.zh`／`.en` 区分语言，并保留同名 `.md`、`.pdf`、`.tex`。11 月 13 日目录内的 `data-review/`、`training-results/`、`experimental-analysis/` 对应数据与方法、实验执行和结果解释。

历史材料保留原有事实和表达，仅调整位置及引用。`notes/` 保存课程评审和前期研究笔记，`archive/` 保存早期提案。数据、训练配置和运行目录保持原路径。

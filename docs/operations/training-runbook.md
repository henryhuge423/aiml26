# 训练运行手册

2026 年 11 月 13 日更新 · [English](training-runbook.en.md)

当前实验版本为 `expanded_20261113`。四组训练及开发评价已完成；数据规模与结果见[进度总览](../progress/2026-11-13/progress-overview.zh.md)。以下命令在项目根目录 `team_project/` 执行。

## 环境、配置与输出

`tools/training_python.cmd` 使用项目内 `.envs/training` 的 Python 与 CUDA 环境。文档图表和 PDF 使用默认 Python、Matplotlib、Pandoc、XeLaTeX 及 Poppler。

| 位置 | 用途 |
|---|---|
| `data/prepared/expanded_20261113/` | 联合冻结清单、发布信息和划分核验。 |
| `configs/expanded_20261113/` | 原数据、扩展无退化、30% 和 50% 退化四组配置。 |
| `runs/expanded_20261113/<组名>/` | 权重、日志、验证与开发预测及结果汇总。 |
| `runs/expanded_20261113/pipeline/status.json` | 实验执行状态。 |
| `docs/progress/2026-11-13/` | 双语报告、LaTeX、PDF、图表与报告核验。 |

四组共用 EfficientNet-B0 初始化、随机种子 20261113、320 像素输入、最大批量 16、AdamW 学习率 0.0001 和 1,200 次优化更新预算。每 300 次更新计算共同模型验证集的指标。`best.pt` 按配置中的来源 AUROC 平均值选择，`last.pt` 保存预算结束权重。扩展无退化组的两者分别来自第 300 和 1,200 次更新。

## 检查完成状态

```powershell
Get-Content runs/expanded_20261113/pipeline/status.json
tools/training_python.cmd tools/verify_expansion_delivery_20261113.py
python tools/verify_expansion_analysis_20261113.py
```

完整核验需要本地数据、权重和逐图预测。仓库中的结果 JSON、图表与报告可用于查看汇总数字。检查项涵盖冻结清单哈希、四组更新数、共同开发图像身份、统计结果及报告链接。

## 执行与恢复

本轮统一入口为：

```powershell
& ./tools/start_expansion_experiments_20261113.ps1
```

该入口在后台隐藏窗口运行，日志写入 `runs/expanded_20261113/pipeline/`。驱动程序检查 GPU 恢复预检与各组完成状态；已完成训练的组跳过参数更新并重新计算开发评价。中断组从其 `last.pt` 恢复。预算结束权重的开发评价由 `tools/run_budget_end_review_20261113.py` 单独生成。

权重绑定配置、初始化、数据清单和训练代码哈希，恢复时核对这些输入。下一轮修改采样、增强或预算时，建立新配置和输出目录，并重新做与新配置匹配的预检。

## 报告构建

```powershell
python docs/progress/2026-11-13/data-review/build_pdfs.py
python tools/report_expansion_training_20261113.py
python tools/report_expansion_analysis_20261113.py
python tools/build_progress_overview_20261113.py
python tools/verify_documentation_20261113.py
```

报告生成程序只更新对应报告与图表。进度索引和文档导航单独维护。交付前检查中英文一致性、图表单位与样本数、每一页 PDF 和保留的 LaTeX 源文件。

旧 `formal_v2` 操作记录见[历史运行手册](../historical/2026-11-09/training-runbook.md)；`formal_v3` 与 EyeQ 的训练结果见[11 月 12 日报告](../progress/2026-11-12/训练进展报告.md)。

# Eye Sentinel

[English](README.en.md) | 中文

Eye Sentinel 研究如何结合眼底照片的青光眼筛查分数和图像质量分数，将需要人工判断的照片送交复核。项目使用两套疾病分类模型和一套质量模型，在相同复核数量下比较不同复核方法。

## 当前进度

截至 2026 年 11 月 11 日，两套疾病模型、原质量模型及加入 EyeQ 标注的质量模型均已完成 20 轮训练。疾病模型的训练集有 10,354 张图片，模型验证集有 1,190 张。修正后的 1,230 张开发图片已获得疾病分数和质量分数，正在用于研究人工复核规则。

两套疾病模型在 AIROGS 验证集上的 AUROC 分别为 0.9766 和 0.9831。加入 EyeQ 的质量模型在 117 张 EyeQ 验证图上的三档质量 macro-F1 为 0.811。分类阈值、复核规则和最终测试仍待完成。

在一项配对试验中，179 张开发图片各生成两档模糊和两档暗光版本。使用 0.5 作为展示阈值时，83 张原标签为阳性的图片中，原图有 8 张低于阈值，较强模糊版本有 27 张。完整数据、结果和图表见 [中文进度报告](docs/progress/eye_sentinel_full_progress_20261111.pdf) 和 [English progress report](docs/progress/eye_sentinel_full_progress_20261111_en.pdf)。

## 数据

疾病模型使用 AIROGS-Light、ODIR-5K 和 Kaggle Eye Diseases 图像。质量模型使用这些图像、FIQS 和 EyeQ。数据扩展工作还包括 AIROGS 原图、Chaksu、PAPILA、FIVES、MSHF、RIM-ONE、BEH、Real Fundus 和 HRF。各来源的标签定义、图像关联与当前使用状态记录在[数据扩展报告](docs/degradation-and-data-2026-11-11.md)和[数据集清单](data/DATASETS-1106.md)中。

## 项目资料

- [进展报告目录及最新训练图表](docs/progress/README.md)
- [完整项目目录说明](docs/DIRECTORY.md)

- [当前提案](proposal-1106.pdf)
- [训练状态与数据规模](docs/TRAINING-STATUS-1110.md)
- [实验复核与 EyeQ 处理](docs/EXPERIMENT-AUDIT-1110.md)
- [训练协议](docs/TRAINING-PROTOCOL-1109.md)与[运行手册](docs/TRAINING-RUNBOOK.md)
- [数据扩展与退化试验](docs/degradation-and-data-2026-11-11.md)
- [外部青光眼数据审计](data/audit/data_expansion_20261111/batch_glaucoma_external.md)和[图像质量数据审计](data/audit/data_expansion_20261111/batch_quality_external.md)
- [EyeQ 质量模型训练](runs/eyeq_quality_v1/README.md)

## 目录

- `configs/` 和 `training/` 存放实验配置与训练代码。
- `data/` 存放数据、清单和审计记录。
- `docs/` 存放进度报告、实验记录和协议。
- `models/` 和 `runs/` 存放模型资源与实验结果。
- `tools/` 和 `tests/` 存放数据处理、训练工具及测试。
- `notes/`、`references/` 和 `archive/` 存放背景资料与历史方案。

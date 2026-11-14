# 模型资源

[English](README.en.md) | 中文 · 2026 年 11 月 13 日更新

项目使用 EfficientNet-B0 疾病分类模型、ResNet50 比较模型及 EfficientNet-B0 质量学生。质量教师采用作者发布的 FTHNet 实现与权重。`checkpoints/` 保存初始化与教师文件，`third_party/` 保存上游代码；训练后的权重位于各自 `runs/` 目录。

此前疾病模型与质量模型已完成 20 轮。本轮 `expanded_20261113` 的四组 EfficientNet-B0 各完成 1,200 次更新，保留验证选中权重和预算结束权重。结果与具体评价集见[最新总览](../docs/progress/2026-11-13/progress-overview.zh.md)和[训练报告](../docs/progress/2026-11-13/training-results/training-results.zh.md)。

初始化、教师和训练权重的完整 SHA256 记录在模型登记、配置指纹与运行产物中。权重保留本地；仓库同步训练实现、配置和精选结果汇总。环境与调用方法见[运行手册](../docs/operations/training-runbook.md)，初始模型与教师准备过程见[历史记录](../docs/historical/2026-11-09/model-preparation.md)。

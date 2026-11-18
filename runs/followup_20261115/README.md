# 11 月 15 日三种子训练结果

[English](README.en.md) · [完整结果分析](../../docs/progress/2026-11-15/completed-experiments/completed-results.zh.md)

冻结数据为 `expanded_20261113`。六个方案、三个共同随机种子的 18 组训练及开发评价已完成，每组 2,400 次参数更新，每 300 次保存权重与模型验证预测；具体配置见 [实验计划](experiment_plan.json)。

- [执行状态](pipeline/status.json)：当前运行、完成开发评价的组及失败信息。
- [CUDA 恢复预检](preflight/verification.json)：连续与中断恢复的权重、逐步损失和状态完全一致。
- [已完成结果](comparison.json)：按更新数和来源汇总，并在完成各组后更新。

启动或恢复：在项目根目录运行 `& ./tools/start_followup_experiments_20261115.ps1`。后台驱动按顺序训练和评价，完成组跳过训练，中断组从 `last.pt` 恢复。输入指纹变化会拒绝恢复；运行期间修改训练实现前应安排新的实验版本。

本地 `seed<种子>/<方案>/validation_checkpoints/` 保存完整权重和验证预测；`development/update_<次数>/` 保存开发原图与固定合成压力集结果。`eyeq_development/` 保存另一质量模型的推理，`error_ranking/` 保存早期逐图连接和关联组交叉验证分数。最终测试图像尚未推理。

`completed_analysis/` 保存 18 个选中模型的新复核分数；主报告目录中的 [completed-analysis.json](../../docs/progress/2026-11-15/completed-experiments/completed-analysis.json) 保存三种子汇总、病例抽样区间、质量分层、实际采样次数和输入哈希。

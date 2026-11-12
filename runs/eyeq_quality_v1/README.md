# EyeQ 质量学生训练

本轮从 ImageNet 权重初始化 EfficientNet-B0，共 20 轮。模型输出连续分数 q，同时用两个有序阈值学习 EyeQ 的 Good、Usable、Reject。损失为所有图片的 FTHNet Huber 回归加 0.05 倍仅在 EyeQ 图片上计算的有序分类损失。0.05 是本次预先固定的实验设置。每轮用验证集的同一联合目标选权重，另记录 EyeQ 三等级 macro-F1、混淆矩阵和等级误差。

训练 4,722 张（EyeQ 1,007 张），验证 616 张（EyeQ 117 张）。EyeQ 仅来自官方训练集。保留确认关联与患者组，隔离测试和其他留出组及其直接未确认候选邻居。旧质量数据保留 4,214 张，暂不使用另外 9,575 张。宽松候选不等同于已确认重复。数据组成和监督同时改变，因此与 formal_v3 的差异不能单独归因于 EyeQ。EyeQ 官方测试图片未参与训练、验证或教师缓存。

从 `team_project` 运行 `tools/training_python.cmd -u tools/run_eyeq_quality.py`。已启动的进程由 `latest_launch.json` 记录。`status.json` 记录阶段，`teacher_cache.log` 记录缓存进度。正式训练在真实 GPU 断点检查通过后自动运行，批次和轮次日志在 `quality_student`。再次启动时会检查当前输入指纹，并在存在匹配的 `last.pt` 时恢复。

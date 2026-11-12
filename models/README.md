# 模型准备记录（2026-11-09）

本文保留 11 月 9 日的模型与环境准备记录。后续疾病模型、原质量模型和 EyeQ 质量模型均已完成训练，当前结果见 [训练状态](../docs/TRAINING-STATUS-1110.md) 和 [EyeQ 训练记录](../runs/eyeq_quality_v1/README.md)。

教师源码保留在 `third_party/BasiQA-bb9b92268256c0398a9909a137d750537aa6b479/`，下载包 `third_party/BasiQA-source.zip`。教师权重为 `checkpoints/net_g_226264S4.pth`，已完成 weights_only 安全加载、严格 state_dict 加载和原版 SoftPool CUDA 真实图片前向验证。上游源码未改动。

完整哈希、下载来源和 checkpoint keys 见 [model_registry.json](../data/audit/2026-11-09/model_registry.json)。[environment.json](../data/audit/2026-11-09/environment.json) 是最初 CPU 环境记录；后续实际训练环境见 [training_environment.json](../data/audit/2026-11-09/training_environment.json)，教师运行证据见 [teacher_runtime.json](../data/audit/2026-11-09/teacher_runtime.json)。教师可运行不等于学生或青光眼模型已训练。

原 Python 环境保持原样。新增 `.envs/training` 使用 Python3.11、torch2.5.1/CUDA12.4 和 torchvision0.20.1；原版 SoftPool 以本机 CUDA12.6/MSVC 编译，setuptools75.1.0 解决旧 torch 的编译兼容问题。依赖清单为 [teacher-extra-requirements.txt](teacher-extra-requirements.txt)，精确 conda 包和 pip 版本记录位于审计目录。pip freeze 含构建机 file URL，只是环境记录，不能直接当可移植安装锁文件。

运行 smoke：在工作区根目录执行 `team_project/tools/training_python.cmd team_project/tools/smoke_fthnet.py --image team_project/data/external/fiqs/images/00000.PNG --output team_project/data/audit/2026-11-09/teacher_runtime.json`。默认固定种子和20次裁剪。该脚本直接加载所选原版架构，并提取执行原版 TargetNet/TargetFC 类体，避免加载无关框架模块；未重写模型计算或替代池化算子。

EfficientNet-B0 官方 ImageNet 权重已下载，哈希和严格加载已验证。仍没有训练完成的质量学生或疾病 checkpoint。后续训练顺序和冻结划分要求见 [训练协议](../docs/TRAINING-PROTOCOL-1109.md)。

后续正式准备已完成：EfficientNet-B0、ResNet50初始化及FTHNet教师完整哈希见 [formal_v2/model_assets.json](../data/prepared/formal_v2/model_assets.json)；2,246条允许质量目标已就绪，三个任务均通过真实GPU优化、验证与精确下一步恢复检查。`runs/preflight/formal_v2/` 里的检查点只用于准备验证，尚未启动正式20epoch训练，也不提供疾病性能。当前入口见 [TRAINING-READINESS](../docs/TRAINING-READINESS.md)。

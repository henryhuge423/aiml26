# 2026-11-10 新版训练运行记录

**当前完成状态：** 队列于2026-11-10 16:44:22 UTC（纽约11:44）完成全部五阶段。A/B各20轮、12,960个batch step；质量学生20轮、15,200个batch step；教师缓存13,789条，质量GPU预检通过。batch step包含AMP跳步，不等于有效优化器更新数。三个任务的best/last、20条epoch日志及run_summary均已存在。本次三路Luna复核、最佳权重验证及EyeQ继续处理见 [本次审计](experiment-audit.md)。已完成训练无需重启；下文启动、恢复与质量门控描述保留为运行历史。

**EyeQ 更新：** 28,792 张原图已完成导入、逐图解码、SHA256 和官方标签匹配。后续独立的 `eyeq_quality_v1` 已完成 20 轮训练，使用 4,722 张训练图和 616 张验证图，其中 EyeQ 分别为 1,007 张和 117 张。详见 [EyeQ 训练记录](../../../runs/eyeq_quality_v1/README.md) 和 [运行状态](../../../runs/eyeq_quality_v1/status.json)。

**本次最终复核：** 65项测试通过。EyeQ全量像素筛查及已知患者/家族闭包完成，确认212条关系，追出2张boundary副本与既有train/val关联。后续开发清单改用 `data/prepared/formal_v3_boundary_repaired_20261110/`（1,230张）；其p_A/p_B/q在 `runs/formal_v3/boundary_development_repaired_20261110/` 已完成，真实验证锚点差均0。原训练/验证/test和checkpoint未改；未锁定灰区、未跑最终测试。下述1,232张为历史冻结规模。

**早间复查与恢复：** A 在第13轮第8077步因原实现未处理 AMP 梯度溢出而停止；已复现、修复并通过42项测试、新版A/B跨epoch GPU证明及真实错误批次重放。纽约08:33从第8000步持久恢复队列。详见 [阶段结果、故障与EyeQ审计](training-review.md)，实际推进以实时状态及step日志为准。下述01:21启动记录是历史快照。

本次按最新数据扩展设计执行，采用独立 `formal_v3`，旧 v2、划分草案、原图及历史工程权重均保留。当前状态以 [队列状态文件](../../../runs/formal_v3/pipeline/status.json) 为准；准备检查不等于完成模型训练。

2026-11-10 06:21 UTC（纽约01:21）已持久启动队列，监督PID43020，首任务疾病A。37项测试通过；独立复核的两项Important问题已修复，A/B跨epoch恢复权重与loss差均为0。后续PID及阶段以实时状态为准。

06:24:39 UTC的 [启动健康快照](../../../runs/formal_v3/pipeline/launch_health.json) 确认：A已完成1轮、进入第2轮，记录到step976；1,190条验证预测身份集合完全匹配model_val，best/last均存在且checkpoint指纹正确。所有记录的loss/梯度和验证分数有限，step连续，已复核放行输入未变化。该快照不是完整20轮完成报告，后续以实时状态和epoch日志为准。

## 数据与配置

| 用途 | 图像数 |
|---|---:|
| 疾病 train / model_val | 10,354 / 1,190 |
| boundary_dev / internal test | 1,232 / 1,233 |
| ACRIMA / HYGD 外测 | 705 / 737 |
| 质量 train / model_val | 12,147 / 1,642 |

保留AIROGS发布者test770张及已纳入的validation769张；一张既有严重技术隔离样本仍不纳入。源内端点分列，已知图像家族/患者组不跨角色；未知患者ID限制保留。17,697个实际图像路径已重新校验SHA256并全像素解码。

配置在 `configs/formal_v3/`：A为EfficientNet-B0，B候选为ResNet50，质量学生为EfficientNet-B0。batch16、workers4、prefetch2、每epoch重建loader、20epochs、AdamW lr1e-4、CUDA AMP；两个疾病模型同预算。所有正式训练均从固定ImageNet权重初始化，工程预检权重不用于正式初始化。

formal_v3 原质量模型使用冻结 FTHNet 的 20 裁剪均值作为教师目标，共 13,789 条，包含 2,246 条重绑定目标和 11,543 条新增目标。EyeQ 的三档人工监督随后用于独立的 eyeq_quality_v1 训练。

## 顺序队列和故障处理

运行顺序：疾病A20轮 → B候选20轮 → 补齐教师缓存 → 质量GPU训练/验证/跨epoch恢复预检 → 质量学生20轮。各模型独占GPU，后续阶段会自动衔接。任一阶段非零退出即停止，状态记为failed，错误和日志保留；不会继续使用失败阶段的产物。

监督进程每10秒更新 `runs/formal_v3/pipeline/status.json`；每任务的 `steps.jsonl`、`epochs.jsonl`、`last.pt`、`best.pt` 和验证预测在 `runs/formal_v3/<任务>/`。Windows Job Object使监督进程退出时清理训练及加载子进程；中断后可从最近完整checkpoint恢复。锁阻止第二个同名队列运行；恢复会检查完整配置/输入指纹。

从项目根目录续跑队列（已有队列运行时会拒绝重复启动）：

```powershell
& team_project/tools/start_training_pipeline.ps1
```

查看进度：

```powershell
Get-Content team_project/runs/formal_v3/pipeline/status.json
Get-Content team_project/runs/formal_v3/glaucoma_a/steps.jsonl -Tail 1
Get-Content team_project/runs/formal_v3/glaucoma_a/epochs.jsonl -Tail 1
```

独立首轮复核见 `runs/formal_v3/pipeline/review.md`；本次AMP修复复核和迁移记录见 `data/audit/recovery_20261110/`，当前真实GPU证据在 `runs/preflight/formal_v3_amp_recovery/verification.json`。旧证据保留但不替代当前代码证明；质量证据由队列随后产生在 `runs/preflight/formal_v3_quality/verification.json`。

首次在沙箱内启动的后台进程没有存活，且没有训练step/checkpoint；随后在持久后台执行环境启动成功。自动审计若需重新启动队列，应使用允许进程在命令结束后继续存活的执行环境；仅获得PID不算开训，必须核验心跳和实际step。不会自动关闭系统睡眠；长训练期间需保持电脑开机和接通电源。

本次不运行测试集性能评估或灰区选择。B仍是候选；需实际模型验证表现支持后才能称可靠对照。模型固定后，再用boundary_dev选择边界、锁定规则，最后进行独立测试。

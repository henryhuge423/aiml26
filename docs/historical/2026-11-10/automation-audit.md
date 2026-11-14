# 自动化审计 2026-11-10

**后续更新：** 本页为纽约13:31的历史快照。EyeQ随后于14:01完成28,792张原图的导入、全像素解码和SHA256；最新三路Luna复核、状态修复和全量关联筛查见 [本次实验审计](experiment-audit.md)。下文“尚未完成导入”不再是当前状态。

## 已完成训练

formal_v3 队列于 2026-11-10 16:44:22 UTC 完成五个阶段。A、B候选、质量学生各20轮；步数分别12960、12960、15200。教师缓存13789/13789完成，质量GPU预检通过。两位Luna子代理分别只读审计训练和EyeQ包。

model_val最佳AIROGS AUROC：A约0.97660（epoch15，0起始），B约0.98313（epoch17）；质量最佳验证loss约0.0005168。此为验证集结果，不是最终测试性能。无需重启已完成训练。下一研究阶段为固定模型、boundary_dev分析与规则锁定，随后独立测试。

## EyeQ导入继续

D:/downloads/diabetic-retinopathy-detection.zip，88310951633字节。外层中央目录可读，15项中包含train.zip.001–005、test.zip.001–007；未发现外层路径问题。此检查不证明内层CRC或像素完整性。

旧导入器等待下载超时。使用既有start_eyeq_import.ps1于18:30:49 UTC重新启动PID5256，提升权限复核确认进程存活，18:31:11 UTC状态进入hashing_archive。普通沙箱Get-Process不可见此提升权限进程；直接重启被文件锁拒绝，因此没有第二个导入任务。stderr为空。

导入输出D:/datasets/EyeQ_20261110/originals；源ZIP保留。C盘临时空间约90GB，可容纳约41.9GB train分卷及10GB余量；D盘可用约143.7GB。导入器先校验源哈希，再选择性提取28792个官方标签图像，全像素解码并计算哈希、精确重叠统计。最终以data/audit/eyeq_acquisition_20261110/import_status.json及import_verification.json为准。当前尚未完成导入或放行；training_released=false。近重复及患者家族审计仍须完成后才能生成新训练版本，不覆盖formal_v3。

验证：本次运行test_eyeq_import.py，3项通过；复核三个run_summary、末轮epoch记录、队列完成状态及质量预检。未修改训练实现或现有数据划分。

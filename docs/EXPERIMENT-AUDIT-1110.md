# 2026-11-10 实验复核与EyeQ继续处理

用户本次要求：检查team-project进度，发现问题修复并继续，使用Luna子代理审计。主代理已派出三路Luna，分别检查训练产物、EyeQ标签/原图和代码可靠性。报告在 `data/audit/session_20261110/{luna_training,luna_eyeq,luna_code}.md`。

**最终状态：** 三个正式模型各 20 轮完成，最佳权重复算、65 项测试和逐条产物验收通过。关联审计发现两张边界图与 train/model_val 图像相同，修正后的开发集为 1,230 张，已完成 p_A/p_B/q 推理。EyeQ 原图导入和关联筛查完成，随后独立的 `eyeq_quality_v1` 也完成了 20 轮训练。正式训练验收记录见 `data/audit/session_20261110/final_verification.json`，新版质量训练见 [运行状态](../runs/eyeq_quality_v1/status.json)。

## 已确认的训练进度

formal_v3于纽约11:44完成五阶段。A和B各20轮、12,960个batch step；质量学生20轮、15,200个batch step。教师缓存13,789条完整，质量预检通过。A的早间AMP故障已恢复；旧异常日志保留，不能将其当成当前失败。batch/global step含被GradScaler跳过的批次，不能全部称为有效优化器更新。

model_val按预定规则选中的AIROGS AUROC：A 0.97660（第16轮）、B候选0.98313（第18轮）；质量最佳验证Huber loss约0.0005168（第16轮）。这是选模开发集结果；质量loss只代表拟合FTHNet教师。B在其他来源不全面优于A，仍保留候选资格和来源标签语义，尚无锁定边界或最终测试成绩。

主代理已逐模型验证checkpoint指纹、完整20轮和连续step、有限模型/优化器状态，并读取best权重重新运行model_val，选模指标与日志完全一致。A/B/质量学生分别有4/1/0次日志明确记录的AMP跳步。原run内的 `validation_predictions.csv` 每轮覆写，表示最后一轮，不能当作best预测；本次对应最佳权重的预测保存在 `data/audit/session_20261110/selected_validation/`，并按路径与SHA256关联得到1,190条 `model_val_pq.csv`。研究训练产物未覆盖；`completion_verification.json` 的 `passed=true`。

## 已修复的问题

1. EyeQ导入状态跨运行合并，继承旧timeout说明；最终 `audited` 每100张更新，最后92张未补记。已加入两个实际临时状态/小ZIP回归测试，先复现失败，再修复通过。原图完整性由28,792条导入清单、官方身份集合和完整解码证据闭合；这是状态记录问题。原状态在本次审计目录备份后纠正。
2. 质量GPU预检目录存在但未完成时，旧队列跳过预检重跑，而预检程序又拒绝覆盖目录，导致重试阻塞。现先严格验证既有证据；无效证据保留到唯一备份后重跑，新的证据仍须通过原验证。三个回归测试先失败再通过；本次已完成训练没有触发该问题。
3. README、训练状态和运行手册仍描述早间运行阶段，已更新为完成状态，历史记录保留时间范围。

完整测试命令：`tools/training_python.cmd -m unittest discover -s tests -v`（从team_project执行）；本次最终65项通过，日志为 `data/audit/session_20261110/tests_65.log`。早期51项及各增量阶段日志保留，避免改写历史审查输入。修改涉及状态记录、编排重试、关联审计、边界修正和只读推理工具；训练数学、数据、优化器和既有权重保持原指纹。

Luna对补丁做了独立静态复核，报告为 `data/audit/session_20261110/patch_review.md`。已保留旧编排review gate，在新gate中绑定重试修复、测试及审查证据并记录lineage；30项输入哈希全部匹配。没有为此次复核重启已完成的正式训练。

## EyeQ 处理范围（当时记录）

本节记录导入和关联筛查阶段。此后 `eyeq_quality_v1` 已完成训练，具体数据和运行结果见文首链接。

下载包 `D:/downloads/diabetic-retinopathy-detection.zip` 为88,310,951,633字节，原包保留。选择性导入目录 `D:/datasets/EyeQ_20261110/originals`。官方train/test为12,543/16,249，总计28,792，逐条标签/角色/grade匹配、所有路径存在，原导入全像素解码与SHA256已完成。与formal_v3无字节级重复，官方患者前缀和精确内容没有跨官方split。

主代理已继续CPU全量关联筛查：EyeQ加历史已解码路径共78,047；ROI pHash半径6、八方向（旋转/镜像）与同名候选，严格pHash/SSIM/ORB-RANSAC联合确认，并将官方患者双眼、既有家族边和SHA相同路径纳入传递闭包。未确认候选保留审阅队列，其已知组件从自动新训练纳入中隔离。完整闭包最终统计以 `data/audit/session_20261110/eyeq_overlap/final_summary.json` 为准；`summary.json` 仅为像素配对阶段，实时像素阶段在同目录 `status.json`。

特征提取和候选生成已完成，共1,106,981对、46,716个唯一端点。原192张LRU缓存会导致反复解码，已保留完整封口证据，使用 `tools/continue_eyeq_confirmation.py` 接续相同候选集。灰度缓存为384×384 uint8，实际约6.89GB，放在工作区审计子目录；先核验源文件size/mtime、候选SHA和封口总数，再一次性解码缓存。SSIM整数乘积/双精度矩计算用于低成本筛选，阈值附近及可能确认的配对仍回到原SSIM/ORB实现；八方向、无关图、近阈值噪声夹具与原确认结果一致。构建过程中发现Windows NumPy默认int32的大小乘积溢出，已用Python `math.prod` 修正并通过实际6,888,554,496字节回归。源ZIP、原图及正式训练产物均未改动。

最终闭包由 `tools/finalize_eyeq_overlap.py` 独立重建，还补入历史工作清单中的全部已知患者边（按来源隔离命名空间，不连接UNKNOWN）和FIQS两种分辨率的同图边。它核验筛查输入绑定并在计算前后复核消费文件哈希，输出 `full_components.csv`、`eyeq_quality_split_proposal.csv` 和 `final_summary.json`。草案在官方train内按组件确定性90/10留出，继承已关联的现有train/val；关联boundary/test、官方split冲突或未确认候选的组件隔离；官方test始终保留test。这些提案字段不是训练放行。

Luna另对官方train固定种子抽取12张原图（每个官方grade各4张）做技术目视检查，样本表、接触图及报告分别为本次审计目录中的 `eyeq_train_grade_samples.csv`、`eyeq_train_grade_samples.jpg`、`luna_eyeq_visual.md`。12例缩略图均呈眼底彩照，出现不同曝光、反光和模糊表现；没有重新评级、临床诊断或由抽样推断全量可用。官方test没有用于这次目视抽样或模型性能调优。

此检索不证明任意裁切、旋转或未知患者关联完全排除；不能据文件名或哈希近似直接继承原图质量等级。EyeQ保持 `training_released=false`；不覆盖formal_v3，不将DR_grade当青光眼标签，不用官方test生成训练教师目标。后续新质量版本需要在这些关联及候选隔离基础上冻结清单、完成新的教师缓存和人工序数监督预检。最终测试性能仍封存。

## 全量关联结论与已落地的边界修正

本轮从每张EyeQ出发，搜索其与EyeQ及历史源的关联；历史—历史的新全配对检索不在本轮覆盖内，历史边和已知患者关系纳入闭包。78,047路径特征、1,106,981对候选全部完成比较，212对达到严格确认门：EyeQ内部39对、AIROGS跨源48对、Kaggle跨源125对。Luna对其中8对原图做独立目视核验，结构相符，见 `luna_pair_visual.md`；这8对不替代其余自动匹配证据。

EyeQ内部有23条直接跨官方train/test的确认像素边，传递闭包为18个混合组件；与现有开发数据关联的EyeQ官方test图像为19张。另有EyeQ train `27188_left.jpeg` 对应现有AIROGS test图像，已进入隔离草案。后续新质量数据版本必须联合处理既有训练别名及EyeQ官方角色，不能仅保留EyeQ CSV划分就宣称独立质量测试。

已识别的全局关联图仅发现两个现有多角色组件，均涉及boundary：

| 新证据原图 | 排除的boundary副本 | 关联的保留角色 |
|---|---|---|
| EyeQ train `17953_right.jpeg` | AIROGS `EyePACS-DEV-RG-339.jpg` | `EyePACS-TRAIN-RG-2266.jpg`，疾病/质量train |
| EyeQ test `23747_left.jpeg` | AIROGS `EyePACS-DEV-RG-1054.jpg` | `EyePACS-TRAIN-RG-2577.jpg`，疾病/质量model_val |

新匹配SSIM约0.998–0.999并具有一致ORB几何对应。独立Luna重建全部已知关系后未发现train–val或涉及formal_v3最终test的多角色分量；该结论限于已识别关系，不证明所有未知患者/变换均被检出。两个boundary重复图在任何边界成绩计算前排除，不根据预测或疾病表现筛样本。

修正文件：`data/prepared/formal_v3_boundary_repaired_20261110/{manifest.csv,exclusions.csv,release.json}`。它是evaluation-only release，训练入口不会将其当作已放行训练数据。保留1,230张（AIROGS783、ODIR305、Kaggle142），原formal_v3冻结文件不变，全部已知跨角色冲突在修正边界分析中消除。追踪和独立代码复核见 `boundary_conflict_review.md`、`boundary_conflict_reconstruction.json`、`boundary_repair_code_review.md`。

已继续原best模型推理：`runs/formal_v3/boundary_development_repaired_20261110/boundary_pq.csv`。逐条保留身份/路径/SHA/来源/组/疾病真值与p_A/p_B/q；三个模型各16张真实model_val锚点重算差均0，推理指纹、release和输出哈希保存在同目录 `verification.json`。尚未选择分类阈值、灰区或工作量准则，未运行最终测试性能。

EyeQ保守草案为train1,007、val117、quarantine11,419、官方test16,249。大量quarantine来自尚未裁决的宽松哈希候选，不表示11,419张均已确认重复或损坏，也不是最终可用训练规模。候选未达严格确认门的1,106,769对仍保留供后续细分/审阅；本次没有据此删除原图或放行EyeQ训练。

审计缓存恢复的后续入口为 `tools/resume_eyeq_confirmation.py`：对实际导入的实现SHA做额外守卫，5项回归覆盖不匹配拒绝。已执行的 `continue_eyeq_confirmation.py` 按像素证据绑定原样保留；新守卫修复未来恢复窗口，不重写本次执行谱系。

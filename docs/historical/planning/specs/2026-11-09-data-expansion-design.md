# 数据划分、质量监督扩展与本机训练效率

2026-11-09。依据用户本轮要求：明确划分、核查 EyeQ/HYGD/EyePACS-AIROGS-Light-V2、Luna 重复与图像质量审计，以及更大 batch 的实际收益。本文是经过数据核对的下一版设计，不是新的已放行训练版本。现有 `formal_v2` 原件和正式配置未改动，正式长训练未启动。

## 明确的四个数据角色

所有任务首先共用患者/已确认图像家族关联组，之后才划分。双眼、重编码、已确认裁剪副本不得跨角色；不同眼图不应仅因同一患者而合并成一张图。无可恢复患者 ID 的来源保留 unknown，不能声称完全患者独立。

| 角色 | 可以做什么 | 不能做什么 |
|---|---|---|
| train | 疾病 BCE；质量教师回归；有人工等级时附加有序监督 | 引入后续开发/测试相关组 |
| model_val | 早停、架构/超参数选择、质量 loss 权重选择、分类阈值/必要的概率校准 | 参与梯度更新；用最终测试选择模型 |
| boundary_dev | 固定 A/B/质量模型后分析 p–q 正误分布，选择复核边界和工作量权衡 | 回流训练网络或反复修改网络后仍宣称边界独立 |
| test | 模型、阈值和边界全部锁定后一次性评价 | 选择模型、阈值、灰区或质量监督权重 |

当前设计不额外切出第五份校准集：早停与可选校准均属于 model_val 内开发，不能声称其校准表现独立；若采用灵活校准器/大范围调参，在 model_val 内按关联组留子集或交叉拟合。p 首先是模型预测分数，未经校准不能直接解释成临床绝对风险。边界开发集始终单独保留。

## 为什么需要新的划分草案

现有 v2 在 AIROGS-Light-V2 上采用项目重划，而没有保留发布者目录。Luna 的元数据交叉计数发现：发布者 `test/` 的 770 张中，535/74/87/74 张分别进入项目 train/val/boundary/test。该做法产生的是项目自己的内部留出，不能报告为发布者测试集成绩。它没有把项目内部 test 回流项目 train，但与现在希望保留上游划分的目标不同。

正式模型尚未训练；此前工程预检/资源探针确实使用了旧v2允许的train/val，可能包含发布者test图片，因此不把770张描述成“从未被工程流程看到”。短程权重仅保留工程证据，全部不用于正式初始化、模型选择或疾病成绩；当前runs仅有preflight和resource_probe，未运行边界选择或正式测试评价。下一版从官方 ImageNet 初始化重新训练，不恢复这些工程权重。没有证据表明旧项目test中74张的疾病成绩被用于调参；人工/技术审图暴露与模型性能调参分开披露。Light-V2 发布者的 test 本身仍不是原 AIROGS 挑战隐藏测试集，也没有可验证患者 crosswalk。

本次生成了逐图、可检查且**未放行**的 [划分草案](../../../../data/audit/expansion_20261109/split_proposal/proposal.json)，而不是覆盖 v2：

- Light-V2 原 test 770 张整体作为项目 test 的 AIROGS 部分。
- 原 validation 中通过现有技术清洗的 769 张整体作为 model_val 的 AIROGS 部分。
- 原 train 中通过清洗的 7,999 张，沿用固定种子/关联组哈希：最高约 10% 的 785 张作为 boundary_dev，其余 7,214 张 train。选择不依赖预测、疾病成绩或测试错误。
- ODIR、Kaggle 的已冻结组及角色保持；HYGD、ACRIMA 继续整来源外测。所有已识别跨源关联必须服从同一组级隔离，不能逐源独立乱切。

| 来源 | train | model_val | boundary_dev | internal test |
|---|---:|---:|---:|---:|
| AIROGS-Light-V2 | 7,214 | 769 | 785 | 770 |
| ODIR 逐眼明确标签 | 2,185 | 277 | 305 | 330 |
| 去重后的 Kaggle 眼病来源 | 955 | 144 | 142 | 133 |
| 合计 | **10,354** | **1,190** | **1,232** | **1,233** |

另留 ACRIMA **705** 张、HYGD **737** 张。HYGD 为 747 原始路径、737 唯一内容、283 患者/重复连接组，不能当作 737 独立患者。

内部和外部结果分来源报告：AIROGS 的 RG/NRG 是可转诊性，ODIR keyword、Kaggle folder、ACRIMA 标签以及 HYGD 临床 GON 定义不同。主 RG/NRG 源内结果单列 AIROGS 的 770 张；不能简单混成统一临床金标准或跨来源挑最好结果。HYGD 可以做模型固定后的“外部效果验证”；若拿它选模型、调边界或阈值，它就成为开发验证，不能再同时称独立测试。HYGD/ACRIMA 均不用于教师目标拟合。

## 质量模型的数据与监督

质量学生不依赖人工 MOS 才能蒸馏。建议使用全部合格疾病 train 图和既有质量 train 图，所有图先用冻结 FTHNet 生成连续 teacher_q，再回归；疾病 model_val 图的 teacher_q 只用于学生验证。boundary_dev 和任一最终测试的教师预测不能用作学生训练或选择目标；模型固定后的正常 q 推理是允许且必需的。

草案进行了已确认图像家族去重，优先保留有人类 MOS 的 FIQS 版本；这不是患者级“每人只留一眼”。所得质量 train **12,147**、quality model_val **1,642**，共 **13,789** 个目标，其中较既有 2,246 目标新增 11,543 个。新数据角色与缓存必须绑定到新版本；旧缓存不得只改文件名直接当成完整新缓存。当前草案未生成新目标、未放行、不能直接训练。

EyeQ 官方表共 28,792 行：train 12,543、test 16,249；等级是 **0 Good / 1 Usable / 2 Reject**。本地只有 5 张预处理示例，只有一张匹配 test 标签，train 对应图为零。`source.zip` 是代码和样例，并非全量原图。故上述训练数**没有计入 EyeQ**。原图官方入口是 EyeQ 仓库指向的 Kaggle EyePACS 竞赛；本轮未下载完整竞赛包或操作账号/条款。

补齐原图后：保持 EyeQ 官方 test 独立；从官方 train 按可恢复 subject ID/双眼及跨源确认家族留出约 10% quality model_val，其余训练，精确图数以组级清单为准。CSV 的 train/test subject 前缀交集为零，但仍需图像内容去重和来源核查。EyeQ 的 DR_grade 不能当成青光眼阴性/阳性标签。

建议联合目标：

`L_quality = L_Huber(q_student, teacher_q) + lambda_ordinal * L_ordinal(human_grade)`

- 教师回归：所有合格且已缓存 teacher_q 的训练图计算，包括没有 MOS 的 EyeQ 图。teacher_q 是模型目标，不冒称人类 MOS。
- 人工有序监督：仅对有人类 Good/Usable/Reject 标签的图计算，可采用 q 上的两个有序可学习阈值，或共享主干的辅助 ordinal head；连续 q 仍用于后续 p–q 分流。若用辅助 head，应另验 q 是否符合人类等级顺序。
- 可选人工 MOS loss 只对确有人类连续评分的图计算，不从三等级编号伪造连续分数；本轮不默认新增这个第三损失。
- 不把三级硬映射为 0/0.5/1，也不因缺 MOS 屏蔽教师回归。各损失先按有标签样本数归一化，lambda 和来源/质量层采样只在 quality model_val 选择，避免一个大来源完全淹没另一个。
- 质量验证分别报告教师一致性 MAE/相关、EyeQ 人工等级的 macro-F1/混淆矩阵/序数指标，以及 q 对人工等级的顺序关系。EyeQ 用于训练后，不能再把同一批训练图称外部质量泛化证据。

BasiQA 作者说公开 FTHNet 权重训练于 FQS，论文在 EyeQ 上做外部评价。项目实际 checkpoint 已核对 SHA256，存在 `params`/`params_ema`；这些不能证明完整训练身份，也不能排除作者曾用 EyeQ 评价调参。FIQS 学生验证是教师拟合验证，不是教师未见的独立临床质量验证。

## 数据审计结果与覆盖范围

三路 Luna 的证据分别在 [EyeQ](../../../../data/audit/expansion_20261109/eyeq/audit.md)、[HYGD](../../../../data/audit/expansion_20261109/hygd/audit.md)、[AIROGS](../../../../data/audit/expansion_20261109/airogs/audit.md)。本轮实际查看 EyeQ 5 张、HYGD 12 张、AIROGS 12 张；技术审阅不替代眼科诊断或全量可判读仲裁。

HYGD 重新逐文件 SHA256 确认 10 对精确重复，其中跨 patient ID 的别名需要连接成组。AIROGS 本地 9,540 张，v2 纳入 9,538，另外两张严重技术问题已隔离。本轮没有把两个数据集当成“新下载未使用”的数据重复纳入。

EyeQ 现存 5 图与 active manifest 无精确匹配，现有确认边也没有对应关联；对其余缺失原图不能开展内容重复审计。AIROGS/HYGD 与开发来源未发现确认跨源匹配，复用了已完成的 SHA、dHash/pHash、SSIM、ORB 多指标边及本轮核验。未重新做缺失 2.8 万张 EyeQ 的全量近重复搜索；“未检出”不证明患者独立。EyePACS 名称相同说明需要核查来源，不能据此证明 EyeQ 与 AIROGS 是同图，也不能默认完全无关。新增原图后必须对全部训练/开发/测试来源统一重跑精确哈希、镜像/裁切候选、局部特征确认与患者组闭包；有确认测试关联时移除开发副本，测试真值保持原来源定义。

## Batch 的实际收益和限制

本机 RTX3060 Laptop 6GiB，AMP，4 CPU threads，当前正式 loader 为 num_workers=0。一次独立对比中每个 batch 档测同样的 512 张真实训练图，含读图、SHA、变换、前后向和 AdamW，另先做 3 个热身批次。以下不是模型精度试验。

| 模型 | batch4 吞吐 | batch16 吞吐 | batch32 吞吐 | batch16/32 整卡采样峰值 |
|---|---:|---:|---:|---:|
| 疾病 A B0 320 | 25.5图/s | 33.2图/s | 35.4图/s | 2.47 / 4.07 GiB |
| 疾病 B ResNet50 320 | 29.4图/s | 32.9图/s | 32.9图/s | 2.68 / 4.09 GiB |
| 质量 B0 384 | 14.6图/s | 19.1图/s | 20.1图/s | 3.10 / 5.30 GiB |

batch32 三模型均通过有限 loss/梯度及参数更新，但质量模型只剩约 0.7GiB 总卡余量，且比 batch16 仅多约 5% 吞吐；ResNet50 几乎无额外收益。推荐先以 **A/B 都 batch16、质量 batch16** 做稳妥的正式候选；不以占满显存作为目标。当前主要瓶颈是同步图片校验、解码和预处理。

另用真实 DataLoader 在 Windows 上测同样的 512 张、batch16、pin_memory=True、prefetch_factor=2（有 worker 时）：

| 模型 | workers0 | workers2 | workers4 | workers4 对 workers0 稳定吞吐比 |
|---|---:|---:|---:|---:|
| 疾病 A B0 | 28.7图/s | 48.7图/s | 100.4图/s | 3.50倍 |
| 质量 B0 | 20.6图/s | 48.3图/s | 90.8图/s | 4.40倍 |

这是另一轮对比，其 baseline 受缓存、后台任务和短程波动影响，不与上表不同轮 baseline 混算倍率。计时排除了进程启动和3个热身批次；workers4 这部分分别耗54.6秒/20.5秒，不能直接用稳定倍率折算整个20轮训练。ResNet50 的多worker吞吐没有单独实测，不能把A的100图/s当B实测。两种模型的多worker峰值 GPU allocated/reserved 与 workers0 基本相同；CPU内存和启动成本增加。

下一轮正式实现建议 batch16 + num_workers4，内存/后台负载紧张时先用2。`persistent_workers=True` 只在同一个 DataLoader 生命周期内保活；如果每epoch重建loader，仍有每epoch启动成本。若重用loader，则要显式向worker同步epoch以维持逐图/epoch增强。当前 engine 仍固定 num_workers=0，本轮只验证独立探针，不能把试验设置误认为已部署，也不能据此沿用旧恢复一致性结论。

扩大 batch 会减少每 epoch 优化器更新次数，因此相同20 epoch的速度提升不保证相同收敛或最终性能。A/B 应采用一致的 batch/样本预算；不自动将学习率乘 4/8，不据资源探针的疾病 loss 选择模型。并行加载还必须保证每图/epoch 随机变换、采样序列、恢复 RNG、源文件哈希验证和患者/家族隔离不退化。质量与疾病模型顺序训练，教师离线打分，不要求同时占用GPU。

原始可复现数据在 [batch sweep 脚本目录](../../../../runs/resource_probe_20261109/README.md)。这些是新增工程探针，不替代正式配置变更后的实际 forward/backward/validation/save/resume 和准备门验证。

## 后续执行边界

本轮交付是审计、逐图划分草案、质量联合监督设计和吞吐证据。旧 v2 的准备完成只适用于原配置/原缓存；本草案 research_training_released=false，不借用旧 readiness。将草案做成新正式版本时，需生成/验证新教师目标、实现并验证有序监督与 loader 配置（若启用）、绑定新输入指纹、独立复核和真实GPU断点验证。EyeQ 在原图缺失时不能声称完成其正式训练准备。

灰区函数/复杂度、边界工作量权衡准则以及最终比较器资格必须在最终测试前写入锁定记录；当前没有训练后错误分布，不能先编造边界。模型训练可以先做，boundary_dev 只在模型固定后使用。对每个测试队列，固定边界先产生 k，再由 B 无真值地选不确定性 top-k；比较灰区外总错误 E/(N-k)、FN 数量及其明确分母，并按患者/关联组重采样。主比较仍是 A+q 对 B+不确定性；同 A 不确定性消融作为补充。

一手参考：[EyeQ 作者仓库](https://github.com/HzFu/EyeQ)、[AIROGS 主办方](https://airogs.grand-challenge.org/data-and-challenge/)、[AIROGS-Light-V2 发布页](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/data)、[HYGD v1.1.0](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)、[FTHNet 作者仓库](https://github.com/HudenJear/BasiQA)、[作者 FQS/FTHNet 论文](https://www.nature.com/articles/s41598-025-24423-8)。

# Team project 审计与开训准备（2026-11-09）

**后续正式准备进展：** 本文保留原审计问题和阶段记录。多批Luna深审计、家族级技术隔离修正、`formal_v2`冻结清单、完整教师缓存及三个任务的真实GPU训练/验证/恢复检查已完成。当前准备结论和执行入口为 [TRAINING-READINESS](TRAINING-READINESS.md) 与 [TRAINING-RUNBOOK](TRAINING-RUNBOOK.md)；本文下方的“尚缺”等描述属于之前阶段，不代表这些项当前仍未完成。

结论：**技术清洗清单与教师 GPU 环境已准备，但尚不能启动可信的正式训练。** 当前已完成文件完整性审计、候选标签/分组清单、Luna 图片抽样、FTHNet 原版 CUDA 前向验证，以及疾病 backbone 初始化权重验证；尚缺近重复/可判读性复核后的冻结划分和正式训练入口。没有产生已训练学生、青光眼模型或独立性能结果。

最新路线以用户完整说明为准：FTHNet 蒸馏质量模型，多来源数据训练青光眼模型，在验证集分析错误/正确样本的 p–q 分布后确定人工复核灰区；边界锁定后应用独立测试集，实际复核比例由边界产生，再与另一个可靠验证的分类模型按不确定性 top-k 复核进行同数量比较。主要指标为剩余自动结果的总体错检率，另列青光眼漏检。详细定义、比较范围和未定参数见 [训练协议](TRAINING-PROTOCOL-1109.md)。1106 review 中“同一分类器为主要比较、测试匹配单列探索”是当时建议，已由本次用户指定的完整方案主比较取代；同分类器消融仍是隔离质量贡献的补充建议。

## 实际执行与范围

- 只读扫描 `data/sources/`、`data/review/`、`data/external/`，**49,256 个物理图片文件、49,246 个 SHA-256 内容**；对全部文件执行严格完整像素解码，不启用截断图片容错。这个数量包含非眼底、质量数据双分辨率、示例和隔离图片，不是训练规模。
- 所有四个现存外部 `source.zip` 重新执行 ZIP CRC 检查，均通过。FIQS MD5 与登记官方值一致；HYGD 官方 750 条 SHA256 重新核验均通过。这证明下载一致性，不保证其中每张图片可用。
- [逐图清单](../data/audit/2026-11-09/image_manifest.csv)、[完整性汇总](../data/audit/2026-11-09/inventory_summary.json)、[下载包检查](../data/audit/2026-11-09/archive_checks.json)、[元数据核对](../data/audit/2026-11-09/metadata_checks.json)。
- [Luna 视觉报告](../data/audit/2026-11-09/luna/audit.md) 和带样本 ID 的拼图/清单提供视觉证据：共 221 张不同路径图片，含首轮 125 张和四个青光眼训练来源各 24 张的补充抽样；19 张联系图均实际查看，部分高风险图追加原尺寸检查。抽样不外推总体低质比例，也不能代替眼科专家判读或疾病真值。
- 原始图片、标签和历史报告未删除或移动；新增证据和工作清单与原始数据分开保存。

## 数据集下载和使用准备

| 来源 | 当前核验结果 | 对正式训练的处理 |
|---|---|---|
| ACRIMA | 705 图，396 glaucoma / 309 normal | 视盘中心裁剪可保留，不因缺少周边视网膜直接淘汰。暂预留来源测试候选，仍须排查跨源相似图。 |
| EyeAugmented | 全类别 15,324 图；目标文件夹 2,325 glaucoma / 2,615 normal | 增强家族未恢复，不能随机按图片分集；非眼底类别和其他疾病不自动转成阴性。 |
| Kaggle Eye Diseases | 全类别 4,213；目标文件夹 1,005 glaucoma / 1,074 normal | 标签和来源沿袭待核查，重点排查 ODIR/其他来源的重编码副本。 |
| ODIR | 7,000 逐眼图，workbook 提供 orig_id 和逐眼诊断文字 | 使用工作簿逐眼匹配；suspected glaucoma 暂不当确认阳性，其他疾病暂不当阴性。 |
| AIROGS-light-V2 | 9,540 图及元数据全部匹配；train 8,000、val 770、test 770 | 标签是 RG/NRG；患者 ID 未提供，原有分集不能据此声称患者独立。 |
| HYGD 1.1.0 | 747 图，548 GON+ / 199 GON−，288 记录患者；校验通过 | 有 10 组精确重复、737 独立图像内容，患者和精确重复合并得到 283 个关联组；建议暂整体作外部测试候选。 |
| FIQS/FQS | 2,246 个原图 ID，各有 1024 版和原尺寸版；2,246 条人工质量记录 | 不是 4,492 个独立样本。低 MOS/reject 应保留为质量训练目标；原尺寸 00135.PNG 截断，1024 对应图可用。 |
| EyeQ | 12,543 train + 16,249 test 条标签；仅 5 个原图示例，另 1 个说明图文件 | 完整原图缺失，不能计为已下载训练图。按当前 FTHNet 路线，它是可选扩展质量数据，不应让其阻断已具备数据的疾病 baseline 准备。 |
| 其他 EyeWorld / FrontofEyeDx / review | 前节/网页目录 4,886 图；FrontofEyeDx 2,321；历史冲突隔离 22 | 独立保存，默认不纳入眼底青光眼监督训练。 |

原有 EyeWorld/FrontofEyeDx 下载来源的完整原始包已不全；本次可重算本地清单和哈希，不能重新证明所有上游原图覆盖率。其历史下载与清理证据仍见 `data/audit/current/` 和 `data/audit/source_archive_validation.json`。

## 需要优先处理的具体问题

1. **FIQS 一个原尺寸 PNG 实际截断。** `external/fiqs/full_size_images/00135.PNG` 严格解码失败，SHA256 `be90421647d76a9c2ff560d223d32ec481bd082959f6c892c701efc648a1ce37`。已核对它与 ZIP 中对应成员逐字节一致，重新解压同一个包不会修复；`images/00135.PNG` 的 1024×1024 版本可完整解码。优先统一使用健康的 1024 版本，并记录该原尺寸文件为不可用，不偷偷开启 LOAD_TRUNCATED_IMAGES。证据见 [定点复核](../data/audit/2026-11-09/targeted_findings.json)。
2. **HYGD 精确重复跨患者 ID。** 10 组中 6 组跨编号，所有这些重复对的疾病标签一致。关联患者编号至少包括 `{65,81,110,180}`、`{187,197}`、`{239,257}`；不能按 Patient 独立随机切分或对重复图重复加权。保留原 Patient，另用 `related_split_group` 联合隔离、按 hash 计独立样本。见 [精确重复清单](../data/audit/2026-11-09/exact_duplicates.csv)。
3. **ODIR 辅助 CSV 的路径错配。** `00_Info/full_df.csv` 共 6,392 行，全部行的 path basename 与 filename 不一致，示例 `0_left.jpg` 对应 path 却指向 `0_right.jpg`；它的 orig_id 口径也不能直接替代逐眼工作簿。候选清单只按 workbook 的 filename 读图、逐眼 Diagnostic_Keywords 映射及 orig_id 分组。仍须核查原始临床标签沿袭，不能把“路径修正”当作“标签已正确”。
4. **近重复尚未完成核验。** equal-dHash 筛出 5,731 组候选，其中 814 组跨来源；包括 FIQS 自己的双分辨率以及大量眼底圆形构图的哈希碰撞。它们**不是 5,731 组已证实重复**。需图像配准/更细特征相似性和目视确认，尤其是增强、裁剪、跨源复制；不能直接全部删除，也不能据当前精确去重就宣称无泄漏。见 [候选清单](../data/audit/2026-11-09/perceptual_duplicate_candidates.csv)。
5. **图像形式和技术质量。** 自动规则标出短边 <224 的 1,781 文件、宽高比 >2 的 656 文件；规则会命中正常视盘裁剪和说明图，只作复核线索。Luna 发现来源中混有前节图、文字/水印/拼图，以及失焦、曝光/遮挡和增强边缘问题。FIQS 低质图属于质量任务的有效样本，不能跟非目标模态一起“清掉”。见 [自动复核候选](../data/audit/2026-11-09/review_candidates.csv) 和 Luna 报告。

## 模型与计算环境

- FTHNet 作者仓库已下载，revision `bb9b92268256c0398a9909a137d750537aa6b479`。源码 ZIP SHA256 `a5d71ee031e9379191cf6807d93ebb1d3e8976acb923f6cc8df13d9577737b2c`。
- 教师 `models/checkpoints/net_g_226264S4.pth`：138,433,897 字节；SHA256 `24335d54083f003a0b7fb011169ef06d1c4967aca0f807bad1ac0240aa35df03`；`torch.load(weights_only=True)` 成功，有 params/params_ema，params 包含 230 个张量。此哈希为本地可复现记录，不是作者提供的独立真实性校验。
- 本地作者 test YAML 指向 FTHNet4 和 params；实际预处理为 512 resize、384 随机 crop、可翻转、ImageNet normalize、默认 20 次采样。训练标签代码除以 full_score=100，不能把网络原始输出再误当 0–100。教师输出还要经过 TargetNet，不能把超网络的参数字典直接当 q。
- 最初审计的 Python 3.12.12 / CPU PyTorch 保留为历史环境快照 `environment.json`。后续建立 `.envs/training`：Python 3.11.18、torch 2.5.1/CUDA12.4、torchvision 0.20.1，真实 GPU tensor 运算通过；见 [训练环境](../data/audit/2026-11-09/training_environment.json)。
- SoftPool 原版 CUDA 扩展已编译。首次构建因新 setuptools 改用 `call`、绕过旧 torch 对 `spawn` 的编译参数注入而失败；固定 setuptools 75.1.0 后解决，没有修改上游池化源码。使用 MSVC14.44 和本机 CUDA12.6，torch 提示与其 CUDA12.4 存在次版本差异；实际 CUDA 算子公式校验和教师前向通过。
- [教师实际运行证据](../data/audit/2026-11-09/teacher_runtime.json)：严格加载 params，在真实 FIQS `00000.PNG` 上执行作者预处理和 20 次裁剪前向，得到有限标量平均值约 0.6292。此单图输出只证明管线可运行，不证明质量模型性能、泛化或临床可用性。教师原始质量训练暴露问题仍未消除。
- EfficientNet-B0 官方 ImageNet 初始化权重最终经 curl 成功获取，21,444,401 字节；SHA256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`，符合上游文件名前缀，已严格加载且前向有限；不是已训练的青光眼模型。之前 PowerShell/Python 的 TLS reset 作为历史失败保留。
- 无已训练质量学生、青光眼 checkpoint、冻结 split、训练配置实现和训练结果。EyeQ 的旧 MCF-Net 不属于本次需要的教师，不能拿它替代 FTHNet。
- 可核查证据：[模型登记](../data/audit/2026-11-09/model_registry.json)、[环境检查](../data/audit/2026-11-09/environment.json)。作者原始代码保留在 `models/third_party/`，未修改。

## 对“高质量还是全部标注数据”的回答

**主训练集选择“所有可靠标注、视盘可判读的质量层”，另设“仅高质量”对照；不用全图质量高低作唯一准入条件，也不无条件使用全部标签。** 高质量筛选会缩窄设备、人群和难例分布，影响现实筛查的稳健性；另一方面，不可判读图会让疾病头被迫拟合缺少可见证据的标签。该判断是本项目的实验设计建议，最终优劣要用同一锁定测试集比较。

疾病损失只覆盖合格疾病标签；不可判读、低质和不合模态图按质量/拒识用途保留并单独评价。测试队列保留实际质量跨度，不仅报告筛后漂亮子集。FTHNet 的全图质量不是青光眼视盘可判读性的真值，HYGD 的 FundusQ-Net 分数也不是人工质量标注。参考 [AIROGS 原论文](https://arxiv.org/abs/2302.01738)，其疾病评价和不可判读检测分开定义。

## 已准备的工作清单与下一步

[逐图候选工作清单](../data/audit/2026-11-09/glaucoma_candidate_worklist.csv) 保留原标签、target_definition、患者组、精确重复关联组、建议用途和未解决原因；[汇总](../data/audit/2026-11-09/worklist_summary.json) 给出计数。ODIR suspected glaucoma 未映射为确认阳性，其他眼病未映射为阴性。所有图的训练可用状态暂为 false，不能把候选条数当最终样本量。

1. 用清单排除非目标/错误图，处理 FIQS 截断版本、HYGD 重复计权和 ODIR 路径问题；对近重复与增强家族做定向复核，补充任务级可判读标注。
2. CUDA 环境、原版教师 smoke 与疾病初始化现已完成；后续在真实训练 batch 上确定显存、batch/梯度累积配置。单图教师 smoke 不替代学生/疾病训练的 forward/backward/resume 检查。
3. 冻结标签语义、来源角色、患者/相关组级 train/val/calibration/test；确认公开教师对质量评估样本的训练暴露。只有来源名称不同不构成独立测试证据。
4. 按训练协议实现训练入口，先真实 batch forward/backward/save/resume，再训练疾病模型和蒸馏质量模型；在验证集分析错误/正确分布并选择灰区边界，锁定后测试，与可靠对照按实际 k 匹配不确定性复核数量。共享 backbone 只作为可选实现实验。

技术清洗输出位于 [data/prepared/1109](../data/prepared/1109/summary.json)：FIQS 只选 2,246 份健康 1024 图像，HYGD 按哈希选择 737 份内容，并保留 10 组跨路径/患者别名。当前文件哈希重新核验，原文件未改。`tools/prepare_clean_manifests.py` 有重复关联、冲突标签、截断原图替代及源文件变更拒绝测试，均通过。清单不自动变成已批准的训练/测试划分。

本次没有启动长训练：近重复、标签语义和可判读复核仍未完成，并且项目目前没有可运行的正式训练入口。环境缺口已补齐，数据和实验设计门槛仍需继续落实。

复现文件审计：在项目根目录运行 `C:\ProgramData\anaconda3\python.exe team_project/tools/audit_training_readiness.py`；下载包检查追加 `--archives`；候选清单与模型证据运行 `team_project/tools/prepare_training_audit.py`（需要本次已下载的教师 checkpoint）。脚本只生成审计产物。

外部核对来源：[FTHNet/BasiQA](https://github.com/HudenJear/BasiQA)、[FQS 数据与论文](https://www.nature.com/articles/s41598-025-24423-8)、[EyeQ](https://github.com/HzFu/EyeQ)、[HYGD 1.1.0](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)、[SoftPool](https://github.com/alexandrosstergiou/SoftPool)。

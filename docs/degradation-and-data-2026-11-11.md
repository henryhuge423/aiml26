# 眼底照片退化试验与数据扩展

## 目前进度

两套青光眼模型和原质量模型各完成 20 轮训练，新版 EyeQ 质量模型也已完成 20 轮。修正后的 1,230 张开发图已评分，分类阈值和人工复核范围待定，最终测试待跑。[训练记录](TRAINING-STATUS-1110.md) · [新版质量模型状态](../runs/eyeq_quality_v1/status.json) · [开发集评分](../runs/formal_v3/boundary_development_repaired_20261110/verification.json)

11 月 11 日早些时候取得 AIROGS 5,000 张、Chákṣu 1,009 张和 PAPILA 488 张。第一轮图像关联筛查后，已分别探索 682、265 和 76 张。PAPILA 的探索子集有 13 张原标签为青光眼的照片，来自 7 人。AIROGS 的 682 张均为不需转诊。[前次数据扩展记录](data-expansion-progress-2026-11-11.md)

## 下载与核验

| 数据 | 本地核查 | 用途 |
| --- | --- | --- |
| FIVES | 800 张原图和 800 条一一对应的质量标签，青光眼、正常、糖网病和黄斑变性各 200 张。青光眼图中 101 张至少一项质量标差，正常图中有 22 张。1,764,974,308 字节，MD5 与 Figshare 一致。 | 优先观察真实低质量病例。按原疾病和质量分层，糖网病及黄斑变性保留原类别。 [官方发布](https://doi.org/10.6084/m9.figshare.19688169.v1) · [本地归档](../data/external/fives/FIVES-v1.rar) · [标签核验](../data/external/fives/inventory.json) |
| MSHF | 官方压缩包 1,136,891,149 字节，MD5 匹配。Original 有 1,302 张，含 52 张青光眼图，其中 50 张在汇总表标为总体质量差。52 张中有 10 张的至少一个质量维度与逐人评分重算的多数票不同。AI-use 归档的训练文件 1,040 张，论文写 1,042 张。 | 保留两版评分，优先核对这 10 张。原图另有 10 对 Local1 字节相同，按同图处理。 [作者论文](https://www.nature.com/articles/s41597-023-02188-x) · [本地归档](../data/external_quality/MSHF_dataset_2.0.zip) · [青光眼逐图质量匹配](../data/external_quality/mshf_glaucoma_quality_join.csv) |
| Real Fundus | 1,098,155,164 字节，ZIP CRC 全部通过。高低质量图各 120 张，编号逐对匹配。SHA256 为 `3ea1624b89c8fbe4563c0e74a11202ff31e4007481bbdbea4494d6332b1d5926`。 | 核对同眼质量分数变化。作者发布未列明单独的数据许可。 [作者发布](https://github.com/dengzhuo-AI/Real-Fundus/releases/tag/v.1.0.0) · [本地归档](../data/external/real_fundus/Real_Fundus.zip) |
| HRF | 18 人、36 张同眼重拍图，较差一张主要有局部或整体失焦。 | 检查模拟模糊与真实失焦的分数变化。 [FAU 官方发布](https://www5.informatik.uni-erlangen.de/research/data/fundus-images/) · [本地归档](../data/external_quality/HRF_allQuality.zip) |
| RIM-ONE DL | 485 张，172 张青光眼。两套划分使用同批图像。ZIP 全成员 CRC 通过。 | 按作者条款和医院划分使用，检查来源差异。 [作者发布](https://github.com/miag-ull/rim-one-dl) · [本地记录](../data/external_glaucoma/ACQUISITION.md) |
| BEH | 634 张，171 张青光眼，ZIP 全成员 CRC 通过。 | 增加达卡医院来源，先核对图像关联和标签定义。 [作者发布](https://github.com/mirtanvirislam/Deep-Learning-Based-Glaucoma-Detection-with-Cropped-Optic-Cup-and-Disc-and-Blood-Vessel-Segmentation/tree/master/Dataset) · [本地记录](../data/external_glaucoma/ACQUISITION.md) |

另取得 KEH/Harvard 官方原包，逐字节核实第三方版本的 1,544 张图都能在官方包找到。官方包自身也有四对图片分别列为正常和早期青光眼，已登记冲突。G1020 第三方镜像取到完整标签表和 70 张图。FunPiQ 取得 300 张局部质量标注图，其中 100 张来自 EyeQ，另 200 张对应 BRSET 和 mBRSET 原照片。BrG 作者入口本次无法打开，Drishti-GS1 官方入口需要账号。[KEH 审计](../data/external_glaucoma/keh/AUDIT.md) · [获取记录](../data/external_glaucoma/ACQUISITION.md) · [FunPiQ 作者发布](https://zenodo.org/records/21838047) · [FunPiQ 本地归档](../data/external_quality/FunPiQ_v1.zip)

## 配对试验结果

从 1,230 张真实开发图按来源和原疾病标签抽了 179 张，每个关联组取一张。83 张原标签为阳性、96 张为阴性。每张在内存生成两个强度的高斯模糊与两个强度的暗光，共 716 个版本，分别用冻结的模型 A 和原 FTHNet 蒸馏质量模型评分。高斯模糊半径以 512 像素长边为基准，分别为 2 和 5 像素。暗光使用 0.70 和 0.40 的亮度乘数。强度为本次试验指定，展示阈值为 0.5。

| 版本 | 阳性标签中分数低于 0.5 | 阴性标签中分数达到 0.5 | 质量分数变化中位数 |
| --- | ---: | ---: | ---: |
| 原图 | 8 / 83 | 1 / 96 | 0 |
| 较轻模糊 | 14 / 83 | 12 / 96 | −0.253 |
| 较强模糊 | 27 / 83 | 7 / 96 | −0.438 |
| 较轻暗光 | 12 / 83 | 2 / 96 | −0.065 |
| 较强暗光 | 20 / 83 | 1 / 96 | −0.214 |

较强模糊让 21 张原本在阈值之上的阳性标签照片掉到 0.5 以下，其中 16 张掉到 0.1 以下。较强暗光让 13 张掉到 0.5 以下，其中 12 张掉到 0.1 以下。值得在 FIVES 和 MSHF 真实低质图上检查质量分数能否抓住这些低疾病分数的病例。

真实配对也用原质量模型评分。较清楚的照片在 Real Fundus 的 120 对中全部得分更高，质量分数的配对差中位数为 0.605。HRF 的 18 对里有 16 对得分更高，中位数为 0.042。两批重拍的差幅不同，可分别用来标定不同强度范围。真实配对仍待与模型训练数据做图像关联核查。新版 EyeQ 质量模型可用同样的真实配对另行比较。[合成配对分数](../runs/degradation_pilot_20261111/paired_scores.csv) · [真实配对分数](../runs/degradation_pilot_20261111/real_pairs_quality.csv) · [分数分布](../runs/degradation_pilot_20261111/distribution.png) · [原图与模拟示例](../runs/degradation_pilot_20261111/example.png)

## 怎么用这些数据

1. **先看真实低质病例。** 优先核对 FIVES 和 MSHF 青光眼图的质量标注，保留来源、设备、眼别、患者和原标签。全图做跨来源比对，将同眼、同患者、裁切版和重编码版分组。按疾病和质量交叉分层，分别标注视盘与杯盘缘可见性；疾病标签以原始临床检查或可判读的同眼照片为依据。
2. **用同眼照片标定退化强度。** 先分开测试失焦、暗光、眩光和低对比，再测试常见组合。用 HRF 及 Real Fundus 观测真实重拍的清晰度、曝光与模型分数差。控制模拟的分辨率和相机域，按视盘区和全图分别检查。需要模拟严重遮挡时，额外标记是否还能判读。用临床标注和图像指标选择合理强度范围。[眼底成像退化模型](https://arxiv.org/abs/2005.05594) · [青光眼影像质量标注指南](https://pmc.ncbi.nlm.nih.gov/articles/PMC12207309/)
3. **用真实开发图选简单复核规则。** 按关联组分折，比较按疾病分数不确定性复核，与加入质量分数后的复核。可先试低质量单独触发复核，以及随质量下降放宽疾病分数复核范围。各方案每折复核同样数量，分别记录漏掉的阳性、误报、自动结果错误率和送审量。加一个同一 A 模型只用疾病分数的对照，另外按协议选择可靠的 B 模型对照。用患者或关联组计算区间，按来源检查结果后锁定规则。
4. **最后才做筛查比例场景。** 以真实样本上的条件错误率为输入，明确设定假设的阳性比例和照片质量比例，比较每 1,000 张会有多少张送审、多少例仍漏掉。需要人口层面的实际分布时，使用目标场景中连续入组、含劣质图和重拍记录的样本。


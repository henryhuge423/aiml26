# Proposal-1106 数据登记

**历史登记，核查日期为 2026-11-07。** EyeQ 原图随后已取得，独立质量模型已完成训练。当前状态见 [数据与实验进度](../docs/degradation-and-data-2026-11-11.md)。11 月 9 日全量解码发现 FIQS 原尺寸 `00135.PNG` 截断，以及 HYGD 10 组精确重复（其中 6 组跨患者编号），详见 [AUDIT-1109](../docs/AUDIT-1109.md)。

核查与获取日期：2026-11-07。保留原始下载包；下列计数不是去重后的训练规模，也不代表标签经临床验证。使用或再分发前确认各自数据使用条款。

| 数据 | 本地位置和状态 | 来源、用途和限制 |
|---|---|---|
| 原有 ACRIMA、EyeAugmented、Kaggle Eye Diseases、ODIR | `sources/eyeworld/`；历史盘点 `audit/current/` | proposal 的原始计数和清理后计数不一致。ODIR 双眼分组与单眼标签口径须重新生成，近重复和增强家族尚未完成。 |
| HYGD 1.1.0 | `external/hygd-1.1.0/`，已解压，原包 `source.zip`；747 JPG，官方 SHA256SUMS 750 项全部通过 | [PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)，548 GON+、199 GON−，288 患者；按 Patient 分组。质量分数由 FundusQ-Net 生成，不是人工质量标注。ODC Attribution 1.0，按来源引用。微调和独立外部测试用途须分开。 |
| EyeQ | 登记时已获取官方代码和标签，train 12,543、test 16,249 条；11 月 10 日取得全部 28,792 张原图 | [官方仓库](https://github.com/HzFu/EyeQ)；原图及其与已有数据的关联审计见 [EyeQ 训练记录](../runs/eyeq_quality_v1/README.md) 和 [实验复核](../docs/EXPERIMENT-AUDIT-1110.md)。 |
| AIROGS-light-V2 | `external/airogs-light-v2/` 已下载解压；ZIP CRC 检查通过；每类 train 4,000、validation 385、test 385，共 9,540 张，另有 metadata.csv | [Kaggle v4](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2)，RG/NRG 为可转诊性标签，不能直接当 glaucoma/normal；页面许可证标为 Other，须核查来源使用条款。原分割的患者独立性未核实。 |
| FQS / FIQS | `external/fiqs/` 已下载解压；原包 9,738,610,740 字节，MD5 与官方值 `1ea6772f3163e0e1f7df8fabd76ac659` 一致；`images/` 和 `full_size_images/` 各 2,246 PNG，另有 metadata 和 30 个分割表 | [figshare v1](https://figshare.com/articles/dataset/FIQS_Dataset_Fundus_Image_Quality_Scores_/28129847)，2,246 图，三等级和 0–100 人工 MOS；CC BY 4.0。[论文](https://www.nature.com/articles/s41598-025-24423-8)说明两个分辨率；须避免同图跨分割，作者指出不同眼可能跨集合，患者独立性需另行核查。 |

## 注意

- EyeQ 原图来自 [Kaggle 竞赛数据](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)，按官方质量标签清单匹配并完成导入。
- 原 proposal 的 10,747 图是原始单标签计数，ODIR 的 1,124 对是另一筛选口径，不可相加。提案中 AIROGS 的数量实际是每类数量，且下载版本验证/测试各为 385，不是 384。
- `sources/` 中的数据、清洗历史和旧审计保持原位；`external/` 为新下载。禁止在未确认许可前将数据上传公开仓库。

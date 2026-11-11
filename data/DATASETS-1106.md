# Proposal-1106 数据登记

**2026-11-09 更新：** 本文为下载时登记；后续全量解码发现 FIQS 原尺寸 `00135.PNG` 截断，以及 HYGD 10 组精确重复（其中 6 组跨患者编号）。完整的新审计、模型下载状态和训练限制见 [AUDIT-1109](../docs/AUDIT-1109.md)。下载校验通过不能替代图片可用性与独立性检查。

核查与获取日期：2026-11-07。保留原始下载包；下列计数不是去重后的训练规模，也不代表标签经临床验证。使用或再分发前确认各自数据使用条款。

| 数据 | 本地位置和状态 | 来源、用途和限制 |
|---|---|---|
| 原有 ACRIMA、EyeAugmented、Kaggle Eye Diseases、ODIR | `sources/eyeworld/`；历史盘点 `audit/current/` | proposal 的原始计数和清理后计数不一致。ODIR 双眼分组与单眼标签口径须重新生成，近重复和增强家族尚未完成。 |
| HYGD 1.1.0 | `external/hygd-1.1.0/`，已解压，原包 `source.zip`；747 JPG，官方 SHA256SUMS 750 项全部通过 | [PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)，548 GON+、199 GON−，288 患者；按 Patient 分组。质量分数由 FundusQ-Net 生成，不是人工质量标注。ODC Attribution 1.0，按来源引用。微调和独立外部测试用途须分开。 |
| EyeQ | `external/eyeq/` 已获取官方代码和标签，train 12,543、test 16,249 条；原始图像未取得 | [官方仓库](https://github.com/HzFu/EyeQ)；标签对应 EyePACS 原图，需另行获取并按官方流程处理。仓库代码标注 CC BY-NC-SA 4.0；图像和标签使用条款须分别确认。EyeQ 和 AIROGS 的可能重叠须排查，未证实存在或不存在。 |
| AIROGS-light-V2 | `external/airogs-light-v2/` 已下载解压；ZIP CRC 检查通过；每类 train 4,000、validation 385、test 385，共 9,540 张，另有 metadata.csv | [Kaggle v4](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2)，RG/NRG 为可转诊性标签，不能直接当 glaucoma/normal；页面许可证标为 Other，须核查来源使用条款。原分割的患者独立性未核实。 |
| FQS / FIQS | `external/fiqs/` 已下载解压；原包 9,738,610,740 字节，MD5 与官方值 `1ea6772f3163e0e1f7df8fabd76ac659` 一致；`images/` 和 `full_size_images/` 各 2,246 PNG，另有 metadata 和 30 个分割表 | [figshare v1](https://figshare.com/articles/dataset/FIQS_Dataset_Fundus_Image_Quality_Scores_/28129847)，2,246 图，三等级和 0–100 人工 MOS；CC BY 4.0。[论文](https://www.nature.com/articles/s41598-025-24423-8)说明两个分辨率；须避免同图跨分割，作者指出不同眼可能跨集合，患者独立性需另行核查。 |

## 注意

- EyeQ 仓库有示例图，不能误认为完整 EyePACS 图像库；原图通过 [Kaggle 竞赛](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) 获取需账户和接受竞赛规则，再按标签匹配。未下载原图，也未验证账户访问。
- 原 proposal 的 10,747 图是原始单标签计数，ODIR 的 1,124 对是另一筛选口径，不可相加。提案中 AIROGS 的数量实际是每类数量，且下载版本验证/测试各为 385，不是 384。
- `sources/` 中的数据、清洗历史和旧审计保持原位；`external/` 为新下载。禁止在未确认许可前将数据上传公开仓库。

# 第二批外部数据质量审计（2026-11-11）

## 范围与结论

仅核查 FIVES、MSHF、Real Fundus、HRF 同眼质量对、FunPiQ。此轮完成本地文件枚举、Real Fundus ZIP CRC、四个已解包来源对 formal_v3 的精确 SHA-256 比对；**图像近似关联筛查未完成**。因此以下用途建议是风险分层，不是放行声明。机器可读计数见 [evidence JSON](batch_quality_external_evidence.json)。

| 数据 | 本地可见内容 | 质量标签/配对 | 初步建议 |
|---|---|---|---|
| FIVES | `external/fives/extracted` 1,602 文件，其中 1,600 PNG；旁有 `FIVES-v1.rar` 与 `Quality Assessment.xlsx` | 目录名提示 800 原图加 800 分割图；Excel质量评分表存在。尚未核对图像与表格主键、标签矛盾或同眼关系 | 质量模型校准候选，**待核对配对和许可**；不可先入训练 |
| MSHF | 解包目录 2,615 文件，2,604 张图（2,602 JPG、2 PNG），有两个评分工作簿及 AI-use 子目录 | 有 `MSHF_quality_scores.xlsx` 和 `Individual_scores.xlsx.xlsx`；多人分数适合分析一致性，但本轮未解析表格/检查同眼及矛盾 | 质量校准候选，保留原始分数与来源；研究用途/再分发条款未确认，先隔离 |
| Real Fundus | `Real_Fundus.zip`，1,098,155,164 bytes；243成员，其中240图像，尚未解压 | 不能从名字推断存在质量标签或患者配对；ZIP CRC通过 | 外部青光眼分析候选仅在解包后证明有诊断标签、可复现授权及去重后评估；当前隔离 |
| HRF | `hrf_same_eye_quality_pairs` 36 JPG，文件名按 1–18 编号组成 18 个 `_bad` / `_good` 名义配对 | 结构明确表达同编号质量对，但是否确为同一原眼/只改变质量未验证。成对图不能作为独立样本随机分割 | 适用于成对质量排序/鲁棒性实验候选，不可直接作真实质量标签校准；来源授权需确认 |
| FunPiQ | 300 PNG 标注加 README（共301文件）；均为掩码而非原始眼底照片 | README 说明标签值 0背景、1 good、2 usable、3 bad；掩码按文件 stem 对应 EyeQ、BRSET、mBRSET 原图。它是像素级质量标注，不应把掩码当眼底图。部分源为受控数据集，须沿用上游条款 | 可用于像素级质量监督/解释性分析；没有匹配原图不能用于图像级真实质量校准或青光眼外部评估；暂隔离 |

## 精确审计状态

- 解包图像数量与格式已通过 PowerShell 文件枚举得到；FIVES 1,600 PNG 包含图像和分割掩码的判断仅为结构推断。
- FIVES、MSHF、HRF、FunPiQ 解包图像对 `data/prepared/formal_v3/manifest.csv` 的 SHA-256 精确交集为 0（逐个图像文件计算哈希）。Real Fundus 虽然 ZIP CRC 通过，但仍未解包，故尚未对其中 240 个图像做精确交集。
- 没有基于 pHash/嵌入的近似图像关联筛查，故裁剪、缩放、重压缩以及来自同一病例的图像关联未知。
- Real Fundus ZIP 共 243 项（240 图像），`zipfile.testzip()` 通过。FIVES RAR 的 CRC 未验证；MSHF、FunPiQ 与 HRF 是目录内容，本轮未逐个解码验证完整性。
- 未解析两份 MSHF 评分表与 FIVES `Quality Assessment.xlsx`，因此逐图映射、一致性、相互矛盾标签数量、病人/眼分组均为未知。HRF 文件名编码的 18 对是名义配对，不代表已证实同眼。
- 上游许可和用途需按来源分别确认。FunPiQ README 指明上游 EyeQ、BRSET、mBRSET 和论文引用要求；对受限来源按其各自条款处理。仅有本地文件不构成再分发或训练许可。

## 面向后续数据决策

1. **真实质量模型校准**：优先继续审计 FIVES、MSHF 的逐图评分映射与标注来源；HRF 限于成对偏好/排序，不充当真实分级标签。FunPiQ 只有在取得相应原图且确认许可后才适合作像素级监督。
2. **青光眼外部分析**：Real Fundus 当前只列候选，需核实原始诊断定义、患者独立性、许可及 SHA/近似重叠后再评估。FIVES、MSHF、HRF、FunPiQ 本轮没有被证明具有适用的外部青光眼诊断标签。
3. **隔离**：在 CRC、SHA 交集、近似关联、标签映射与许可证均完成前，五个来源都不并入 formal_v3 或用于外部结果。FunPiQ 掩码与其源图必须始终分开管理。

## 与既有审计的交叉核对

本轮没有重复两项先前已完成的逐图工作。data/external/fives/inventory.json 已验证 FIVES 的 800 张原图与 800 条质量标签逐一匹配，其中 200 张青光眼图有 101 张至少一个质量维度为差。data/external_quality/mshf_glaucoma_quality_join.csv 已将 52 张 MSHF 青光眼图与质量工作簿匹配，50 张总体质量为差，10 张至少一个最终质量字段与三位标注员重算的多数票不一致。两套资料继续隔离，直到完成跨来源近似关联、患者或眼别关联和许可审计。

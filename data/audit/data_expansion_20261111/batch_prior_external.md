# 第三批既有外部候选审计（2026-11-11）

## 范围、口径与结论

本轮只复核 AIROGS 官方 train 的前 5,000 张、Chákṣu train、PAPILA，以及 GFID/RDD-Net 派生的 Harvard 包和其 KEH/Harvard 官方同源包。核查依据为本地归档、清单、哈希绑定和既有审计；没有重新训练、推理，亦没有读取或改动 formal_v3 冻结集合。

**使用结论：四项均不放行给训练、校准、模型选择或独立外部性能报告。** AIROGS、Chákṣu、PAPILA 中的 `no_candidate_found` 仅表示本轮八方向 ROI pHash 筛查未命中，仍带有来源选择偏差、未知患者关系及未排除的近重复风险；它们此前的开发性分数已被查看，不能日后称为未触碰外测。GFID/Harvard 整体隔离；KEH 官方包内的跨类别相同图继续逐文件隔离。

| 来源 | 下载/内容完整性 | 样本与标签语义 | 本轮关联筛查/隔离状态 | 明确使用状态 |
|---|---|---|---|---|
| AIROGS 5k | `0.zip` 只取目标条目所在 range：2,690,844,681 / 9,726,644,528 bytes；每张目标图 CRC、SHA-256、解码和官方清单核对通过。range SHA-256 为 `4b72…6639a0`；不能把它写成完整 `0.zip` 的官方 MD5 核验。 | 5,000：NRG 4,837、RG 163。NRG=无需因青光眼转诊，不等于健康；RG 也不等价于统一临床诊断。无患者 ID。 | 4,318 候选隔离、682 未命中（均 NRG）。RG 的距离<=4候选有 147 条严格边，涉及 105 张 RG；其中 146 条连到既有 Light-V2 RG。另 58 张 RG未证实独立，仍隔离。 | **隔离；不可作独立外测或训练。** 与项目 AIROGS-Light 同源，且大量 RG 已直接连到既有图。 |
| Chákṣu | Figshare v2 只取原图所需 range（1,106,020--1,151,679,702）；1,009 张均逐图 ZIP CRC、内容 SHA-256 和解码通过。range SHA-256 为 `0f96…b047`，不替代整个训练 ZIP 的 MD5。 | 1,009：NORMAL 872、GLAUCOMA SUSPECT 137（专家多数票）。suspect 不是确诊青光眼，也不能与 AIROGS RG 混用。Bosch 文件映射为 145 表行中的 104 个训练文件；41 行不在训练图中。患者/眼别未知。 | 744 候选隔离、265 未命中（NORMAL 236、SUSPECT 29）。距离<=4 的 1,132 候选中，7 条严格/精确确认边都在 Chákṣu 内部；距离6的 6,824 条未二检，保持隔离。 | **隔离；不可作诊断性能或阈值资料。** 未命中部分也已被探索性评分查看。 |
| PAPILA | 完整 `PAPILA.zip`：590,857,635 bytes；现算 MD5 `7d73fc5157d44b221a62ed5a4baf3779`，与官方记录一致。文件级 CRC、图像 SHA-256、解码及清单已记录。 | 488 图/244 人、每人 OD/OS：健康(0) 333、青光眼(1) 87、疑似(2) 68。原始三类语义保留，疑似不得合并为确诊。 | 以患者双眼为整体隔离：412 图/206 人涉及候选；76 图/38 人未命中（0=57、1=13、2=6）。365 条距离<=4候选未有严格确认边；距离6的 2,712 条未二检。 | **隔离；不可把图片或双眼当独立样本，也不可报告敏感度/泛化性能。** 未命中 76 图已被探索性评分查看。 |
| GFID/RDD-Net Harvard 与 KEH 官方包 | 第三方 `Harvard.zip` 124,029,917 bytes；既有记录称其 MD5 与 Zenodo RDD-Net 发布相符。官方 KEH `processed_data.zip` 123,981,705 bytes，现算 MD5 `305f29a31a7db2c9bd5536b9c077a09e`，与 Dataverse 公布值一致。 | 第三方外层 train 清单为 1,234 行（0=630、1=604），但发布页未给出可信 0/1 映射/许可说明；其为视盘裁切版本，不能补作原始病例标签。官方 KEH 目录计数 normal 788、early 289、advanced 467，共1,544，与论文 1,542 不一致；患者/双眼关系未知。 | 第三方 `Harvard.zip` 的 1,544 PNG 与官方包 1,544 图逐项 SHA-256 同源（`third_party_sha256_mapping.csv` 1,544 行），故绝非新队列。官方仅 1,540 个不同 SHA-256，并有四组跨类别同图冲突（下列图）。 | **全部隔离，禁止训练、推理、校准、模型选择和性能结论。** 外层 test 未被推理；必须先澄清标签映射、原始许可、版本差与患者关联。 |

## 可复现证据与复核结果

- 当前哈希绑定仍一致：AIROGS `candidate_manifest.csv`、Chákṣu `originals_verified.csv`、PAPILA `candidate_manifest.csv` 的 SHA-256 分别与各自 `source_binding.json` 记录一致。AIROGS/Chákṣu 的 source binding 是执行后记录，不是发布方签名。
- AIROGS 的 `screened_manifest.csv` 为 4,318 `quarantine_candidate` + 682 `no_candidate_found`；Chákṣu 为 744 + 265；PAPILA 为 412 + 76。上述数字是筛查状态，不是重复图计数或独立性证明。
- PAPILA 的 244 位患者和 OD/OS 均在候选清单中；因此隔离与将来统计必须以患者/关联组为单位。AIROGS 与 Chákṣu 没有可用患者 ID，不能因图片未命中而推断病人独立。
- Harvard 派生训练清单的 1,234 个本地 PNG 与 CSV 行数一致，但这只确认派生清单的文件存在，不能修复标签含义或使之成为独立数据。官方同源映射覆盖的是 Harvard.zip 的 1,544 PNG。

## KEH 跨类别同图：继续逐文件隔离

官方 KEH 包内以下完全相同 SHA-256 的四对同时出现在 normal 与 early 类别；八个目录成员均继续隔离，不能以文件夹类别充当真实标签：

- `early_glaucoma/193.png` ↔ `normal_control/706.png`
- `early_glaucoma/194.png` ↔ `normal_control/707.png`
- `early_glaucoma/271.png` ↔ `normal_control/781.png`
- `early_glaucoma/272.png` ↔ `normal_control/782.png`

该冲突已足以否决这八个文件的监督或评估用途；不应被误读为其余 1,536 个目录成员已经通过患者、近重复或标签审计。下一步若需重新评估 KEH，必须先从发布者或可信原始资料取得标签定义/许可的可追溯说明，再对关联组（而非单图）做跨项目近重复和患者闭包审计。

## 不可外推的既有探索结果

AIROGS 的 682 NRG、Chákṣu 的 265 图、PAPILA 的 76 图曾在固定权重与验证锚点不变的条件下进行探索性评分；三个记录均明确 `classification_threshold_locked=false`、`boundary_locked=false`、`final_test_evaluated=false`、`research_training_released=false`。这些运行证明的是当时复现锚点差为 0，不证明标签可比、患者独立、阈值有效或外部性能。它们不改变本审计的隔离结论。

## 审计定位的文件

- `data/external/airogs_candidate_5k/source_binding.json`、`candidate_manifest.csv`、`screened_manifest.csv`、`positive_secondary_distance_le4.csv`
- `data/audit/airogs_labels_20261111/chaksu_train_labels/source_binding.json`、`originals_verified.csv`、`screened_manifest.csv`、`secondary_distance_le4.csv`
- `data/external/papila_candidate/source_binding.json`、`candidate_manifest.csv`、`screened_manifest.csv`、`secondary_distance_le4.csv`
- `data/external/gfid_candidate/Harvard.zip`、`Havard_train.csv`；`data/external_glaucoma/keh/AUDIT.md`、`third_party_sha256_mapping.csv`

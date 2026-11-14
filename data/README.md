# 数据工作区

[English](README.en.md) | 中文 · 2026 年 11 月 13 日更新

当前疾病训练与评价版本为 `prepared/expanded_20261113/`，包含 24,491 张图像：15,480 张训练、2,046 张模型验证、2,053 张规则开发、3,470 张内部及新增保留测试、705 张 ACRIMA 和 737 张 HYGD 外部测试。数据组成及检查见[联合版本说明](prepared/expanded_20261113/README.md)、[数据审计](../docs/progress/2026-11-13/training-results/data-audit.zh.md)和[进度总览](../docs/progress/2026-11-13/progress-overview.zh.md)。

新增池 `prepared/expansion_20261113/` 含 9,070 张有明确二分类标签的图像，另单列 204 张疑似图像。新增池是数据审查产物；联合版本记录其后续训练、验证、开发与测试分配。文件哈希、RGB 像素哈希、来源标签、已知患者和确认图像关联在清单与审计中保存。

| 目录 | 内容 |
|---|---|
| `sources/`、`external/`、`external_glaucoma/`、`external_quality/` | 原始来源文件及后续下载。 |
| `metadata/` | 来源索引和标签表。 |
| `prepared/` | 按版本冻结的数据清单、发布记录和分组核验。 |
| `audit/` | 来源核对、图像关联、标签和划分检查。 |
| `review/` | 待复核图像及同图异标签记录。 |
| `archive/` | 来源包和历史附属文件。 |

此前 `formal_v3` 疾病模型与原质量模型完成 20 轮，EyeQ 质量模型也完成 20 轮。质量数据包括 FIQS 和 EyeQ；EyeQ 导入覆盖 28,792 张原图，其中部分图像按允许清单参与质量训练。真实低质量阳性在联合疾病清单中分别为训练 72、验证 12、规则开发 12 和测试 50 张，FIVES 与 MSHF 保留各自质量定义。

数据文件、逐图清单和完整审计保留本地原路径。仓库发布来源说明、双语报告、冻结版本摘要及精选结果 JSON。[历史数据布局](../docs/historical/2026-10-22/data-workspace.md)和[早期数据登记](DATASETS-1106.md)保留此前阶段的记录。

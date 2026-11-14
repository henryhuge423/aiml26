# 已审计新增数据池

[English](README.en.md) | 中文 · 2026 年 11 月 13 日

新增池含 9,070 张图像，其中 3,446 张阳性；另单列 204 张疑似图像。本目录记录数据审查完成时的来源、标签、图像关联和文件身份。同日后续的训练与评价用途记录在[联合版本](../expanded_20261113/README.md)：5,142 张用于训练、858 张模型验证、833 张规则开发、2,237 张保留测试。

| 来源 | 图像 | 阳性 |
|---|---:|---:|
| airogs_original_rg | 3 | 3 |
| beh | 333 | 83 |
| chaksu | 861 | 0 |
| crfo | 67 | 40 |
| fives | 384 | 191 |
| fundus_avseg | 42 | 14 |
| grape | 518 | 518 |
| hrf | 30 | 15 |
| keh | 1,529 | 747 |
| lag_part1 | 4,812 | 1,688 |
| les_av | 22 | 11 |
| mshf | 49 | 49 |
| papila | 420 | 87 |

`pool_manifest.csv` 记录来源标签、文件和像素哈希、照片与患者分组及作者划分，`release.json` 绑定审查输入。LAG、GRAPE、FIVES、Fundus-AVSeg 与 MSHF 联合分组；KEH 保留视盘裁剪域。MSHF 的 PNG 训练副本与原 TIFF 解码 RGB 像素一致。疑似图像保留其原始标签，单列于 `suspect_manifest.csv`。

[数据审查报告](../../../docs/progress/2026-11-13/data-review/data-and-degradation-review.zh.md) · [训练数据审计](../../../docs/progress/2026-11-13/training-results/data-audit.zh.md)

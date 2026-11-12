# Eye Sentinel：2026-10-23 proposal 与数据安排

更新：2026-10-23。本文记录当时的外眼图像实验方案；目前的青光眼研究与训练结果见 [项目进度](../docs/degradation-and-data-2026-11-11.md)。

我们研究有限复核额度下的图片选择：每一百张图片只能选出五张、十张或二十张时，怎样找到更多错误预测和未知类别图片？全球健康中的复核能力与计算条件在这里转成实验设置。

从 Wide-Field 选择 3–5 个拍摄方式接近的类别，训练预训练 MobileNetV3-Small。候选包括 GoogleBing-InfectEye 中的 Pterygium 153 张、Blepharitis 99 张、Hordeolum 121 张。其他选定类别用作陌生输入，按整个类别分成开发集和最终测试集。统一标签关系、检查近重复及相关图版后，保存全部分组与划分。

比较模型分数、预测一致性、两者结合三种排序。每个方法使用同一批图片、同一额度。报告错误捕获率、未知类别捕获率、剩余错误和各类别选择比例。分别测试未知图片占比为 0%、10%、30% 的模拟情景，重复抽样并记录波动。

扩展实验比较全部单次预测、全部四个图像版本、只对第一轮难图使用四个版本。在同一笔记本 CPU 上记录时间、峰值内存和模型大小，比较接近计算时间预算下的表现。

离线演示可以打开一批现有图片、设置额度、查看选中的图片、预测及选择原因。四人分别负责数据、模型、评估、演示与报告，交叉核对划分和结果。

## 数据口径

- 类别目录共 34,449 张：Fundoscopy 27,242 张、Wide-Field 4,886 张、FrontofEyeDx 2,321 张。
- 另有 22 张标签冲突图单独存放，全部图像共 34,471 个不同 SHA-256 内容。
- FrontofEyeDx 有 94 个有图目录，含眼前节、视网膜及视神经相关类别；每类 1–98 张，中位数 23 张。
- 文件名、caption 和目录用于核对标签；模型输入照片像素。图中文字及多面板检查属于数据准备。

## 当前材料

- [Proposal v3](../archive/proposal_versions/Eye_Sentinel_Project_Proposal_v3.pptx)
- [Dataset PDF](../docs/historical/Eye_Sentinel_Dataset_Introduction.pdf)
- [Dataset Markdown](../docs/historical/Eye_Sentinel_Dataset_Introduction.md)
- [评分表对照](Eye_Sentinel_Rubric_Alignment_2026-10-22.md)

本次修改前的文件保存在 `../archive/proposal_versions/before_resource_revision_2026-10-23.zip`。

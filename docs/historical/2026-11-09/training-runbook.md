# 正式研究训练运行手册

**2026-11-13 当前训练记录：** 数据和退化审计完成，expanded_20261113四组各完成1,200次有效优化更新。结果见[中文](../../progress/2026-11-13/training-results/training-results.zh.md)／[English](../../progress/2026-11-13/training-results/training-results.en.md)。新入口为`tools/start_expansion_experiments_20261113.ps1`；已完成预算会被核查并保留。下文为旧版本的环境、命令与历史记录。

**Current training record, November 13:** Both audits and all four 1,200-update expanded_20261113 arms are complete. The new entry point is `tools/start_expansion_experiments_20261113.ps1`; it verifies completed budgets and preserves outputs. Results are in the bilingual report above. Commands below document earlier releases.


**2026-11-10当前入口：** formal_v3顺序训练已完成，三个模型各20轮，见 [训练记录](../2026-11-10/training-status.md) 和 [本次审计](../2026-11-10/experiment-audit.md)。已完成任务无需重复启动。下面仅保留v2的历史命令和证据，其旧代码指纹不能替代v3的新预检。

**版本范围：** 本页保留原v2可复现命令。后续 [数据划分与质量扩展设计](../planning/specs/2026-11-09-data-expansion-design.md) 已给出新逐图草案和更大batch/并行加载实测；草案尚未放行，不能直接套用下列命令声称训练的是新版。正式长训练尚未启动。

本手册对应 `data/prepared/formal_v2/release.json` 的 `formal_v2` 数据版本（release格式为 `formal-research-v1`）。最终准备状态以 [TRAINING-READINESS](../../operations/training-readiness.md) 的实际核验结果为准；长训练与准备验证分开记录。旧 `data/prepared/formal/` 已被 `SUPERSEDED_BY.json` 标记为历史版本，训练入口会拒绝启动它。

## 固定的数据版本

| 用途 | 图像数 | 使用规则 |
|---|---:|---|
| 疾病训练 | 9,874 | AIROGS、逐眼 ODIR、去重后的 Kaggle 多来源代理标签 |
| 模型验证 | 1,341 | 早停/最佳 checkpoint 选择；主要选择终点是 AIROGS RG/NRG 源内 AUROC |
| 边界开发 | 1,397 | 训练后做校准、正确/错误 p–q 分布与灰区选择；不进入优化器 |
| 内部锁定测试 | 1,397 | 图像家族隔离的研究 holdout；AIROGS 等未知患者 ID 的局限保留 |
| ACRIMA 外部测试 | 705 | 整体保留的视盘中心裁剪域，未用模型表现筛图 |
| HYGD 外部测试 | 737 | 保留独立内容，按 283 个患者/精确重复关联组记录 |
| 质量学生训练 | 1,794 | FIQS 健康1024版本，监督目标是固定教师20裁剪均值 |
| 质量学生验证 | 452 | 学生对教师拟合验证，不称教师未见的独立泛化 |

数据和分组依据：`data/prepared/formal_v2/` 的 `manifest.csv`、`exclusions.csv`、`release.json` 及 `data/audit/deep/release_inputs/`。冻结文件绑定候选清单、原始清单、技术裁决和重复边哈希。后续修改数据或规则应生成新版本，不覆盖既有版本。v2修正了技术隔离的家族传播：ODIR `1281_right.jpg` 的箭头覆盖及Kaggle副本均被隔离，较v1只减少该Kaggle训练副本，其余保留行和全部外部测试不变。

EyeAugmented 的4,940张二分类候选因父图谱系未恢复而整体留在扩展队列。原始数据均未删改。ODIR 正例保留明确 glaucoma 关键词（可含共病），疑似不作阳性，负例只用纯 normal fundus。AIROGS NRG 不等于健康眼；不同来源的标签定义保留在逐图表中。

## 环境与初始化

- 入口 `tools/training_python.cmd` 使用项目内 `.envs/training`，不依赖原 CPU Python。
- Python3.11、torch2.5.1/CUDA12.4、torchvision0.20.1；SoftPool 原版 CUDA 扩展以本机CUDA12.6/MSVC构建，setuptools75.1.0。
- A：EfficientNet-B0；B候选：ResNet50；质量学生：EfficientNet-B0连续回归头。两个疾病模型用同一训练和验证清单、输入尺寸、增强、采样和预算。
- 初始化和教师 checkpoint 均校验完整 SHA256。教师保持冻结；批处理只补充 contiguous 内存布局兼容，不改权重或池化数学，批20与逐图20裁剪均值差已核验小于1e-4。
- A/B 输入320、质量输入384，batch4、AdamW lr=1e-4、weight_decay=1e-4、20epochs、CUDA AMP、梯度裁剪5；这是初始固定配置，不是经过测试集调参后的结果。
- 疾病图先补黑边为正方形再缩放，保留视盘视野；训练只水平翻转。质量学生保持原图全幅缩放，水平翻转，避免用改变质量的色彩/模糊增强配固定教师标签。

## 运行命令

以下在工作区根目录 `E:\jhu\1-AIML_for_GlobalHealth` 的 PowerShell 中执行，任务顺序运行，共用一块GPU。

若需重建或续算已有允许队列的教师缓存：

```powershell
& team_project/tools/training_python.cmd team_project/tools/cache_teacher.py --manifest team_project/data/prepared/formal_v2/manifest.csv --release team_project/data/prepared/formal_v2/release.json --output team_project/data/prepared/formal_v2/teacher_scores.csv --crop-batch 20
```

正式训练 A、B候选和质量学生：

```powershell
& team_project/tools/training_python.cmd team_project/tools/train_model.py --config team_project/configs/glaucoma_a.json
& team_project/tools/training_python.cmd team_project/tools/train_model.py --config team_project/configs/glaucoma_b_candidate.json
& team_project/tools/training_python.cmd team_project/tools/train_model.py --config team_project/configs/quality_student.json
```

若中断，使用同一配置恢复，例如：

```powershell
& team_project/tools/training_python.cmd team_project/tools/train_model.py --config team_project/configs/glaucoma_a.json --resume team_project/runs/glaucoma_a/last.pt
```

已有 run 的 `last.pt` 会阻止无意从头覆盖。checkpoint 绑定配置、数据版本、初始化和代码哈希；质量任务额外绑定教师缓存及其元数据。恢复模型、AdamW、AMP scaler、CPU/Python/CUDA RNG、epoch 与下一批位置。恢复时保留超出 checkpoint 的旧日志备份，避免重复 step 被误读成新训练。

缓存以SQLite逐行事务为权威记录，CSV/JSON为原子导出。强制中断后可重跑同一命令续算，已提交的教师行不会丢失；修改教师实现、权重、预处理、crop batch或清单会拒绝混用旧缓存。

本次2,246个教师目标最初全部在v1的允许质量队列计算。v2修正仅涉及疾病训练的一张副本，质量队列的sample_id、path、SHA256、split及教师计算指纹逐行完全一致，因此显式验证后重绑定缓存，未重新预测或改变目标数值。来源缓存和两个release的哈希、重绑定证明保存在v2 `teacher_scores.json` 的 `reuse_provenance` 中。不能只改 sidecar 的manifest哈希来复用任意缓存。

## 输出与选择规则

每个 `runs/<任务>/` 包含配置、step/epoch日志、`last.pt`、`best.pt`、验证预测CSV和run summary。A/B最佳 checkpoint 只按模型验证集 AIROGS源内 AUROC 选择；其他来源的指标分列。质量学生最佳 checkpoint 按验证 Huber loss（对教师）选择。验证预测保留图片/哈希/来源/关联组及标签类型，质量教师输出与人工MOS分列。

B当前只是已实现、可训练的比较候选。正式训练后必须核验其验证表现与不确定性定义，才能称为“具有可靠验证依据的对照模型”；ImageNet权重和准备用小批次验证不提供青光眼性能证据。Luna提出的额外性能门槛是审计建议，未自动替代本项目的最终比较方案。

完成模型训练后，用预留边界开发集绘制正确/错误/FP/FN在p–q空间的分布及每格样本数。校准、疾病分类阈值与灰区候选的选择顺序和数据使用须单列记录，必要时在此开发池内再分组交叉拟合。边界选择后锁定，再对同一测试端点运行 A+灰区 与 B+不确定性top-k；k由A的固定边界实际产生。不同来源的临床终点分别评价，不将RG/NRG、文件夹标签和GON混成一个临床真值。

## 已知研究范围

AIROGS/Kaggle/ACRIMA/FIQS没有可恢复的完整患者ID；图像家族分组不能证明所有患者独立。近重复检索是保守的多指标审计，未检出不等于证明没有未知变换或患者重叠。已识别FIQS关联组不进入最终疾病测试；公开教师的完整训练暴露仍不支持宣称FIQS教师未见泛化。Luna检查的是技术质量和来源问题，没有代替眼科专家的临床标签仲裁。

这些限制随研究数据版本保存；它们不应被“训练准备完成”改写成临床性能或部署结论。

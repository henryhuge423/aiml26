# Eye Sentinel team project

## 当前实验（2026-11-10）

**formal_v3 已于纽约时间11月10日11:44完成全部五阶段**：疾病A、B候选和质量学生各20轮，13,789条教师缓存完整，质量GPU跨epoch预检通过。最佳检查点已独立核验并重算model_val预测。本次三路Luna审计发现2张boundary同图变体与train/val关联，已生成独立修正清单：1,232→1,230张，并完成对应p_A/p_B/q开发推理。原训练、验证、最终测试及检查点保持原指纹，65项测试和最终验收通过；灰区和最终测试尚未运行。

- [当前训练记录、数据规模与恢复方法](docs/TRAINING-STATUS-1110.md)
- [本次三路Luna审计、修复与继续处理](docs/EXPERIMENT-AUDIT-1110.md)
- [早间恢复、阶段结果及EyeQ下载/存储审计](docs/TRAINING-REVIEW-1110.md)
- [实时队列状态](runs/formal_v3/pipeline/status.json)
- [独立复核与修复验证](runs/formal_v3/pipeline/review.md)

EyeQ的88.31GB下载包及28,792张官方标签原图完整，官方train/test分别12,543/16,249。全量筛查确认212条图像关联，含48条EyeQ—AIROGS及125条EyeQ—Kaggle关联；发现官方split混合组件18个，EyeQ官方test中19图关联既有开发数据。完整结果在 `data/audit/session_20261110/eyeq_overlap/final_summary.json`。

新版 `eyeq_quality_v1` 使用 EyeQ 官方训练集中的 1,007 张训练、117 张验证，加入人工三档质量监督。旧质量数据中与留出样本有关联或疑似重复的部分暂时隔离。见 [本轮设置与恢复方法](runs/eyeq_quality_v1/README.md) 和 [运行状态](runs/eyeq_quality_v1/status.json)。

后续边界开发使用 [修正清单](data/prepared/formal_v3_boundary_repaired_20261110/manifest.csv) 和 [已生成的开发分数](runs/formal_v3/boundary_development_repaired_20261110/boundary_pq.csv)，不再使用原1,232张全表做边界分析。原始冻结文件保留。

## 历史准备入口（2026-11-09）

最新方向依据用户完整说明：训练青光眼模型与 FTHNet 蒸馏质量模型获得 p、q；在验证集观察正确/错误分布，确定人工复核灰区并锁定；测试复核率由边界实际产生，与可靠对照模型按不确定性复核相同数量图片，比较剩余自动结果的错检率和青光眼漏检。详细定义见训练协议。

- [全量审计与开训准备状态](docs/AUDIT-1109.md)：49,256 个图片文件全像素解码、下载完整性、标签/重复问题与环境缺口。
- [训练准备协议](docs/TRAINING-PROTOCOL-1109.md)
- [最新数据划分、EyeQ联合监督与batch/loader实测](docs/superpowers/specs/2026-11-09-data-expansion-design.md)：三路Luna审计与逐图草案；扩展版本尚未放行。
- [正式训练准备状态与证据](docs/TRAINING-READINESS.md)
- [运行手册与训练命令](docs/TRAINING-RUNBOOK.md)
- [Luna 实际图片质量抽样](data/audit/2026-11-09/luna/audit.md)
- [模型下载和验证登记](data/audit/2026-11-09/model_registry.json)

以下是11月9日的准备证据，未运行v2正式20epoch训练。v2冻结研究数据在 `data/prepared/formal_v2/`，包括家族级去重/技术隔离、来源标签与已知患者分组；其教师缓存和原配置工程预检证据保留历史范围。当前训练使用顶部的v3和新GPU证据。原始图片未改变。未知患者ID、近重复检索覆盖和质量标注的研究范围见准备状态；旧 `data/prepared/formal/` 已弃用。

## 1106 方案与历史审查

[proposal-1106.pdf](proposal-1106.pdf) 是之前的完整提案。下列审查与建议保留作为历史参考；质量模型选择已由上述用户更新具体化。

- [方案审查与建议](docs/proposal-1106-review.md)
- [数据集登记与获取状态](data/DATASETS-1106.md)
- [项目行动清单](docs/NEXT-STEPS.md)

以下 v4 交付物、相关 notes 和 tools 均为历史材料，其模型、重拍和实验细节可能与 1106 方案不同。保留原位置以免破坏旧脚本和链接。

## Historical v4 deliverables

- [Proposal v4](Eye_Sentinel_Project_Proposal_v4.pptx): quality-aware glaucoma screening, retake feedback and expert review. Five slides with English scripts and Chinese notes.
- [Speaker notes v4](Eye_Sentinel_Speaker_Notes_v4.md): English script and Chinese reading notes for rehearsal.
- [Dataset introduction PDF](Eye_Sentinel_Dataset_Introduction.pdf)
- [Dataset introduction source](Eye_Sentinel_Dataset_Introduction.md)

## Folder guide

- `data/`: cleaned image sources, metadata, review samples, audit records, and AppleDouble sidecars. The redundant 4.59 GB EyeWorld ZIP has been removed; its content validation record remains in `data/audit/source_archive_validation.json`.
- `references/`: original project brief and FrontofEyeDx cleaning instructions.
- `notes/`: proposal alignment, related work, and the historical presentation review.
- `tools/`: `build_eye_sentinel_v4.py` rebuilds the v4 deck and the dataset PDF. The PDF reads the Markdown source. Earlier builders and decks remain available for project history.
- `archive/proposal_versions/`: superseded proposal decks retained for project history.

For the image inventory and use limitations, start with [the data workspace guide](data/README.md).

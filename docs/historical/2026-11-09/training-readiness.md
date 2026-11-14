# 正式研究训练准备完成

**2026-11-10当前状态：** formal_v3全部五阶段已完成，三个模型各20轮、教师缓存13,789条完整、质量GPU预检通过；最新65项测试和best权重复算通过。新增EyeQ关联审计发现的2张boundary副本已在独立开发清单排除（1,230张），p_A/p_B/q推理完成；原训练/验证/test和权重保持。见 [最终审计](../2026-11-10/experiment-audit.md) 和 [训练记录](../2026-11-10/training-status.md)。以下ready结论和31项测试仅是旧v2历史证据。

**范围更新：** 本页结论仅对应下述原 `formal_v2` 和batch4配置。用户后续要求的数据扩展、保留AIROGS发布者划分、EyeQ监督及batch/loader优化，见 [最新设计与三路Luna审计](../planning/specs/2026-11-09-data-expansion-design.md)。新逐图草案未放行、教师目标尚未补齐，不借用本页的ready结论；正式长训练仍未启动。

当前数据版本为 **formal_v2**；机器核验 [readiness.json](../../../data/prepared/formal_v2/readiness.json) 的 `research_training_ready=true`。正式20epoch长训练尚未启动。运行环境、数据版本、教师目标和三个训练入口均已通过实际检查，启动命令见 [运行手册](../../operations/training-runbook.md)。

## 已完成的检查

- 多批Luna审计覆盖来源/标签、扩展图片质量、跨来源重复、技术裁决、代码缺陷及最终冻结版本。后续深入目视审阅408个分层样本，另对131项给出第二轮技术动作；保留了此前221张初审记录，不把分层抽样比例当全量质量比例。
- 对12,846对候选执行多指标近重复审计，冻结883条关联边（10条精确、873条强视觉关系）；52对配对拼图被目视查看，其中16条眼底边逐条标记人工确认。自动高置信边与人工确认明确区分。EyeAugmented父图谱系无法恢复，4,940张二分类候选留在扩展队列。
- 冻结清单整合已知患者、双眼、精确重复与确认图像家族。带箭头的ODIR图片及Kaggle副本均隔离；v2较v1只移除该Kaggle训练副本，全部外部低质挑战样本保留。旧版本通过SUPERSEDED_BY标记阻止启动，原始图像未改动。
- [独立Luna释放复核](../../../data/audit/deep/luna_release_review/audit_v2.md)：`PASS_WITH_LIMITATIONS`，未解决issues为空；路径、输入指纹、计数、ODIR/HYGD分组、活跃家族边及质量训练与疾病测试隔离均通过。
- [最后一次独立增量审计](../../../data/audit/deep/luna_final_delta/audit.md) 已复核缓存重绑定、家族隔离修正、过期版本拒绝及总准备门，未发现Critical/Important或准备阻塞项。
- 三个初始化/教师文件均以完整SHA256固定。原版FTHNet与SoftPool在GPU运行，质量队列2,246条20裁剪均值全部有限；缓存逐行绑定图片哈希、split、教师权重、实现和预处理。v1→v2只在允许质量行完全相同、教师计算指纹不变时显式重绑定，预测数值未改变。
- **31项测试通过**，包含未放行/被替代清单拒绝、图像家族跨集合、教师暴露隔离、冲突标签、缓存强制中断恢复、目标错配和checkpoint/RNG恢复。
- [三个任务的真实GPU准备验证](../../../runs/preflight/formal_v2/verification.json) 全通过：疾病A、ResNet50比较候选和质量学生均发生真实参数更新，生成验证预测及best/last检查点。从中断恢复后的下一步与连续运行相比，最大权重差和loss差均为 **0.0**。这些是可运行性证据，不是疾病性能结果。

## 冻结规模

| 数据用途 | 图像数 |
|---|---:|
| 疾病训练 | 9,874 |
| 模型验证 | 1,341 |
| 后续灰区边界开发 | 1,397 |
| 内部测试 | 1,397 |
| 外部ACRIMA | 705 |
| 外部HYGD | 737 |
| 质量学生训练 | 1,794 |
| 质量学生验证 | 452 |

Manifest SHA256：`201378a147fa4d0c54f22fbeb61806b971924738a9cf7b4af502fc67644556c9`。逐图来源、标签定义、关联组与排除原因分别见 `data/prepared/formal_v2/manifest.csv` 和 `exclusions.csv`。

## 启动与核验

在工作区根目录执行首个疾病模型：

```powershell
& team_project/tools/training_python.cmd team_project/tools/train_model.py --config team_project/configs/glaucoma_a.json
```

比较候选和质量学生命令、恢复方式及输出说明见运行手册。重新核查当前准备状态：

```powershell
& team_project/tools/training_python.cmd team_project/tools/check_training_readiness.py
```

## 结果解释仍需保留的范围

研究版本没有声称所有患者身份都已恢复：AIROGS、Kaggle、ACRIMA、FIQS存在未知患者ID；图像家族不跨集合不等于证明全患者独立。近重复检索未检出也不证明不存在未知裁切、旋转或上游重用。FIQS留出用于学生拟合教师验证，不能称为教师未见的独立质量泛化。

疾病监督是保留来源定义的代理标签，RG/NRG、文件夹glaucoma/normal与HYGD临床GON分别报告。Luna的技术质量检查没有替代眼科专家临床仲裁。B目前是已准备好的训练候选，必须在正式训练后的验证结果上通过比较规则，才称为可靠对照；准备检查点不能承担这一证据。

后续顺序保持用户方案：正式训练并选择模型 → 在预留开发数据检查正确/错误p–q分布并锁定灰区 → 独立测试得到实际k → 对照按不确定性复核同样k → 比较剩余错检率与青光眼漏检。最终测试未用于训练、教师目标生成或边界选择。

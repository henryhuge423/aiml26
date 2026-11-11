# Eye Sentinel：已有工作、数据与 proposal 差异

核查日期：2026-10-22。本文保留下载数据前的文献与初步定位记录；其中“未见原图、缺少元数据、来源不明”等判断应以同日完成的数据审计为准。请先阅读 [数据集介绍 PDF](../Eye_Sentinel_Dataset_Introduction.pdf)、[数据集说明](../Eye_Sentinel_Dataset_Introduction.md) 和 [当前审计目录](../data/audit/current/README.md)。整理前审计快照位于 `../data/audit/baseline/data_audit/`。本文的文献对照可继续参考；数据整理记录见 `../data/audit/`。

## 建议的主张

我们希望帮助社区卫生工作者完成有症状眼表问题的首次评估：取得可用照片，结合简短病史，形成下一步建议，并将有用信息交给接诊诊所。

这是一条值得尝试的项目路线。手机识别、多病种分类、多模态和转诊系统均有既有研究；差异应落在选定人群、具体问题、交互流程和评价问题上。下面的研究构成项目基础，不构成全球首次性的系统检索结论。

## 1. 这两个数据源是什么

依据本地 `../references/Project Opportunity_EyeSentinel.pdf`：EyeWorld 与 FrontofEyeDx 是项目提供的网页抓取眼病图像集合，覆盖眼睑、结膜、角膜、虹膜、晶状体、外伤、视网膜、视神经及肿瘤等。材料描述为数百至数千张图像量级，但未分别报告两个集合的确切数量、病例数、类分布、重叠程度及标注流程。

本地 `../data/metadata/frontofeyedx/eye_diseases_list.xlsx` 实读有 101 个疾病条目；课程概述称 100-disease collection。它是一份工作清单，不是已验证的 101 类训练集规模证明。名单也混有不同层级：例如 corneal ulcer 是病变表现，而 fungal keratitis 是病因分类，两者可能出现在同一张图里。

本地 `../references/Instructions to Derm-FrontofEyeDx Database.pdf` 是清理手册及示例图集，不是完整原始图像库。它要求保留真实临床照片、纠正错放类别、拆分多面板图、保留解剖背景，并将难判断的图像交给复核。它说明了数据需要整理，不能据此推定所有图片已经完成这些步骤。

### 分别可以确认多少

| 名称 | 已有证据 | 对项目的含义 |
|---|---|---|
| EyeWorld | 课程将其列为数据源；网上确有 ASCRS 的 EyeWorld 专业出版物，但未找到把课程图像包与该网站逐项关联的数据卡 | 以课程数据包为准；保留图片来源 URL 后再确认具体来源 |
| FrontofEyeDx | 课程概述、清理手册与疾病清单；本轮未找到可核验的公开数据卡、正式论文或独立下载入口 | 可以介绍为课程提供的待整理图像集合，不编造机构、规模和公开 benchmark 排名 |

同名检索还会出现 2026 年 EyeWorld 眼部动态世界模型论文；没有证据表明它就是本课程数据源。

两个集合均可能混合眼外观、裂隙灯及眼底等图像。先识别拍摄模态，再选择任务。普通手机外观照片、专用照明下的眼前节照片和眼底照片应分别建模与评价。

目前查看的 team_project 文件中没有完整原始图像库，也没有逐病例配对的症状、年龄、视力或转诊结局。数量和可做类别需要拿到原图后统计。课程材料将用途定位为研究或非营利；实际图片使用仍需核对来源与授权。

## 2. 代表性已有工作

| 工作 | 已经做到什么 | 对 Eye Sentinel 的启发 |
|---|---|---|
| Ueno 等，BJO 2024 | 6,442 张图像，其中 6,106 张裂隙灯、336 张手机图；将 36 种眼前节疾病归为 9 类，研究诊断与分诊 | 手机眼前节多病种识别已有基础；需要明确自己的使用场景 |
| DeepIK，npj Digital Medicine 2024 | 12 个中心、23,055 张裂隙灯图，区分细菌、真菌、病毒、阿米巴和非感染性角膜炎，含外部和前瞻评价 | 精细感染分类需要可靠诊断标签，不能直接将网页文件夹等同于微生物学标签 |
| Eye Home Clinic，Informatics 2025 卷，2024-12-31 上线 | 用户自行手机拍照，先判断质量，再识别异常，并结合医生复核 | 质量检查、重拍和手机远程评估已有应用实例 |
| JHU / Aravind 等，2025 medRxiv 预印本 | 6,794 张图、2,956 位患者，将眼前节图像与年龄、视力等结合；在 210 位患者中前瞻评估手机端白内障分类 | 轻量、多模态和离线部署可作为技术基础；该引用版本为预印本 |
| IDS / Eye-YOLO，BMC Medicine 2026 | 24,671 张图，含裂隙灯和手机图；整合轻量多病种识别、图像质量判断与紧急程度分类，并经微信小程序评价 | “手机 + 质量控制 + 分诊”本身已有接近的系统 |
| Peek CEH，Lancet Digital Health 2021 | 肯尼亚 36 个社区单元的集群随机试验，结合手机决策支持、短信提醒与实时报告研究眼科服务使用 | 项目最终可以评价患者是否到诊；转诊闭环也有可借鉴的既有研究 |

这些论文的设备、病例、目标和评价方式不同，不直接比较各自准确率来判断模型优劣。

## 3. 推荐展开的三个差异

### A. 围绕首次评估安排提问与行动

选择有症状的眼表问题作为起点，重点关注需要及时进一步评估的角膜异常。工具依次帮助工作者：取得可用照片、记录少量相关病史、查看图像发现、决定是否重拍或请求复核、生成转诊摘要。

有吸引力的研究问题是：在固定的问诊时间内，哪些问题最值得问？初期可用医生共同设计的问卷/规则搭建流程；后续有真实配对数据时，再比较图像单独、固定问卷和按已有信息选择问题。可以评价漏掉的重要病例、额外复核比例、问题数量和完成时间。

课程概述本身已经建议“照片 + 问题 + 转诊”。所以单说三者结合属于落实课程愿景；若要更具体的贡献，应提出自己的问题选择方式、工作流或验证实验。

### B. 从教学网页图片走向可迁移的模型

把两个集合用成一个数据研究问题：模型学到的是病变，还是网页文字、拼图样式及来源特征？

先保留原始文件，生成去重簇、来源和清理记录。建立一个人工核验的固定测试集，比较训练数据清理前后效果；另做文字/边框遮盖敏感性分析。不能只在充满诊断文字的测试图上报分数。

若两个数据源均有足够共同类别，且核实来源与近重复图后，可以做 A 训练/B 测试，再反向测试。若元数据充足，优先按原网站/原病例分组；若只有重复图片簇，只能称图像簇隔离，不能称患者隔离。

报告类别敏感度、macro F1、置信度校准，以及低质量图片上的表现。人为模糊、曝光和压缩测试有助于观察脆弱性；真实手机场景泛化需要另外的手机数据。

### C. 从评估结果走到可用的转诊交接

病例摘要包含照片、症状、时间线、图像发现、复核原因及接诊所需信息。与接诊诊所共同决定哪些字段最有用。

课程期可以交付有案例的交互原型，评价工作者和接诊者能否理解、信息是否完整、是否节省整理时间。临床合作阶段再观察转诊完成率和等待时间。网页图像数据本身不能验证这些结局。

## 4. 在现有材料上可做的任务

| 任务 | 需要什么 | 建议 |
|---|---|---|
| 数据审计与可追溯清理 | 原图、来源、重复检测与人工复核 | 最先做；形成每类可用数量和质量记录 |
| 眼前节/眼底/非临床内容路由 | 人工标注一批模态与内容类型 | 作为数据处理组件，不必当主模型 |
| 少量外观类别的基线分类 | 有足够样本且标注可解释的类别 | 按审计结果选择 3–5 类；避免一开始承诺全病谱 |
| 层级/多标签建模 | 重新整理诊断、病变与部位关系 | 避免把“角膜溃疡”和“真菌性角膜炎”强行当作互斥标签 |
| 跨来源测试 | 来源可追溯、共同类别、去除跨集合重复 | 最适合课程内形成研究结果的方向 |
| 图像质量/需复核预测 | 专门标注质量和可判读性 | “非临床图片”和“临床但拍得差”分开处理 |
| 图像与症状融合 | 同病例的真实配对图像、临床信息及参考标签 | 后续研究；虚构病史只能演示界面流程 |
| 病变检测或分割 | 新增框或像素标注 | 疾病文件夹标签本身不足以训练定位模型 |
| 端侧原型 | 已训练模型、量化和目标设备测量 | 报模型大小、实际延迟和内存，不能只凭网络名称称其手机可用 |
| 转诊摘要及随访原型 | 设计的病例和真实用户反馈 | 可以与图像模型并行推进 |

优先路线：原图审计 → 选定眼表子集 → 基线分类 → 清理/跨来源实验 → 质量提示与病例摘要 demo。

## 5. 可放入演讲的英文

Prior studies have shown that smartphones can support eye disease screening. We want to build on that work for the first assessment in community care. Eye Sentinel would help a health worker capture a usable photo, ask a few relevant questions, and prepare a clear summary for the receiving clinic. We would start by testing what our image model learns from the two web-sourced collections, and whether it holds up when the image source changes. That gives us a practical first step toward the workflow we hope to build with clinics.

PPT 第 2 页适合增加“cleaned data and cross-source testing”；第 3 页用一个就诊案例串起照片、问题和摘要；第 5 页将课程图像实验与临床工作流评价衔接起来。5 分钟主讲只选一个技术研究问题，相关工作可放附录。

## 来源

本地：`../references/Project Opportunity_EyeSentinel.pdf`、`../references/Instructions to Derm-FrontofEyeDx Database.pdf`、`../data/metadata/frontofeyedx/eye_diseases_list.xlsx`。

- Ueno et al., BJO 2024: https://pubmed.ncbi.nlm.nih.gov/38242700/
- DeepIK 2024: https://www.nature.com/articles/s41746-024-01174-w
- Eye Home Clinic: https://www.mdpi.com/2227-9709/12/1/2
- JHU / Aravind multimodal cataract preprint: https://www.medrxiv.org/content/10.1101/2025.11.14.25340278v1.full
- IDS / Eye-YOLO 2026: https://link.springer.com/article/10.1186/s12916-026-04905-w
- Peek CEH 2021: https://pubmed.ncbi.nlm.nih.gov/34167763/
- Peek community volunteer algorithm 2020: https://pubmed.ncbi.nlm.nih.gov/32558656/
- Multimodal bacterial/fungal keratitis study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11758206/
- ASCRS EyeWorld publication description: https://www.ascrsmarketing.org/
- EyeWorld site copyright notice: https://www.eyeworld.org/legal-notice/
- Same-name EyeWorld world-model preprint (not established as this dataset): https://arxiv.org/abs/2603.14039

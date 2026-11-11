# Eye Sentinel 演示文稿审阅

审阅日期：2026-10-22。本文是对旧版 5 页演示稿的历史审阅记录，不代表对当前 v3 的重新审阅；当时检查了正文、全部备注和逐页导出图片，并对照 oral presentation rubric、项目机会说明和数据清洗手册。原始稿保存在 `../archive/proposal_versions/`。

## 总体判断

现有版本结构清楚、字体易读、语气审慎，已说明课程原型而非临床成果。保留这五页框架即可。最值得优先修改的是：

1. 第 5 页补足真正的部署路径。目前主要讲离线评估，与第 2 页重复，尚未充分回答 rubric 的 Path to Deployment。
2. 区分课程可交付的图像分类原型与未来手机转诊产品。第 2 页说症状是未来扩展，第 3 页却呈现包含问卷、图像质量检查、紧急程度和转诊摘要的完整工具，容易让听众误判完成范围。
3. 将“专用设备图像与社区手机照片的差异”列为主要障碍。项目说明明确提醒，数据同时包含眼前节和眼底图像，拍摄可能需要专用设备。外眼病种的标签也不保证图片就是普通手机可拍摄的照片。
4. 加入一项解释性的视觉证据。当前五页全部由文字和编号构成，排版整齐，但还没有让听众看见数据是什么、输出是什么。

## 按 rubric 检查

| 评分维度 | 目前表现 | 最有价值的补充 |
|---|---|---|
| Problem Definition，5 分 | 用户和资源受限场景清楚；具体筛查范围只在备注中出现 | 将“优先研究照片可见、可能危及视力的部分病变”放进第 1 页；说明具体类别以数据审计为准 |
| AI Strategy / Approach，5 分 | 数据集、基线、预处理、数据划分和指标齐全 | 补充数据审计后才能确定的类别及样本数；说明同一患者、病例、来源和近重复图片如何避免跨集合 |
| Strategy for Clinical / Public Health Use，5 分 | Capture–Assess–Refer 容易理解 | 第 3 页显式标为未来应用流程；展示一个概念输出，说明原型当前能实现的部分 |
| Stakeholders & Considerations，5 分 | 四类利益相关者基本覆盖 | 从一般需求进一步落实到：谁负责复核、谁接受转诊、谁维护工具，以及患者能否实际到达转诊点 |
| Path to Deployment，5 分 | 当前最薄弱：第 5 页仍然主要是模型评估 | 增加课程原型、目标场景外部验证、受监督试点三个阶段，以及每阶段的进入条件 |
| Clarity & Organization，5 分 | 五页顺序合理 | 消除第 2、5 页重复；统一“当前原型”和“未来产品”的表述 |
| Visuals，3 分 | 配色、字号和对齐一致；无明显溢出 | 用真实且获准使用的样本说明数据差异，或用清楚标注的概念输出说明用途 |
| Time Management，2 分 | 现有英文讲稿约 533 词，具备控制在 5 分钟内的条件 | 将排练目标定为约 4 分 30 秒，留出停顿和换人时间；无需为了五页强行每人发言 |

这些是改进方向，不是对教师最终分数的预测。

## 逐页建议

### 第 1 页：Eye Sentinel

保留项目名、社区医务人员这一用户定位和全球负担数据。WHO 2026 年 2 月 10 日的事实清单支持“至少 10 亿人的视力损伤本可预防或仍未得到处理”。这是全部相关视力损伤的全球规模，不是本工具的可服务人群。现有讲稿已作区分，值得保留。

目前“The unmet need”下方仍是较宽泛的眼部初步评估。建议将其具体化为：

> Initial assessment of selected visible eye conditions when specialist care is difficult to reach.

在正文或口头补充：优先考虑可能危及视力的表现，具体标签待审计确定。不要把“识别所有眼病”或“解决 10 亿人的视力问题”作为承诺。

标题较大但没有影响阅读，不必为了改变而重做封面。

### 第 2 页：Data and model plan

这是信息最密的一页。左列换行较多，右列的“macro F1”被拆行。建议删减通用训练细节，把有限空间用于数据可信度。

建议三列保留：

- Curate：screen image types；remove duplicates and diagnostic cues；retain supported classes。
- Train：pretrained MobileNetV3-Small；training-only augmentation；class weighting if needed。
- Evaluate：group related images before splitting；per-class sensitivity and macro F1；review failure cases。

细节放到备注或 appendix。尤其要明确：

- 先归组、后划分；同一多面板图拆出的样本不能因此进入不同集合。
- 普通精确去重不够，还要注意不同裁剪、分辨率或水印版本。
- 若有可靠病例或患者标识，按病例或患者划分；若没有，应说明这一验证局限。
- 来源分组与跨来源测试是处理来源偏差的办法；逐类别看错误本身不能消除来源偏差。
- 缺失或不一致标签可以核查，但学生核查不能被描述为临床诊断验证。
- 不能把“没有出现在本次分类标签中”解释为健康。不要自行补出 normal 类。

目前文件夹只有说明文档、病种表和 PPT，没有原始图像，因此不能确认各类有效样本数、最终类别、配对症状、患者元数据和可用正常对照。展示时将这些标为待审计事项，不编造数据规模。

### 第 3 页：Use in a community clinic

保留三个步骤，但标题建议改为 **Intended clinical workflow**，副标题明确 **Future use, subject to validation**。

必须解决与第 2 页的范围关系：本学期实现图像分类和离线演示；问卷、经临床审核的规则、自动图像质量判断、紧急程度和转诊整合，分别标注为拟议功能。现有资料不足以声称这些组件已经存在。

可将每步的长句缩短，利用腾出的空间放一份概念结果：输入图片、模型候选类别、无法判断时的人工复核提示、供工作人员填写或确认的后续行动。标明“Concept only”，不展示伪造的模型概率或性能。

不要把 softmax 高概率直接称为可靠性，也不要把疾病分类性能直接当作转诊紧急程度的准确性。如果未来真的开发转诊功能，应分别验证。

### 第 4 页：People, trust and safety

现有利益相关者列表可保留，但“clear responsibility”“responsible data use”等措辞比较抽象。建议换成具体任务：

| 相关方 | 希望获得什么 | 需要确定的安排 |
|---|---|---|
| 患者和家属 | 能理解结果、隐私得到保护、能获得后续照护 | 图片使用说明、结果解释、转诊可达性 |
| 社区医务人员 | 工作负担可承受、知道何时请求复核 | 简短操作流程、培训、复核渠道 |
| 接诊机构和眼科人员 | 有用的病例摘要、可管理的转诊量 | 接收方、复核责任、转诊容量 |
| 项目运营方 | 能维护工具并追踪问题 | 数据管理、维护负责人、异常报告 |

主幻灯片不一定容纳完整三列表格，可以只保留每行最重要的两项，细节放到问答准备中。

数据用途限制应明确：项目说明要求基于这些抓取图像训练的模型用于研究或非营利情境。这是项目给出的限制，并不等同于逐张图片版权和许可已经核验，也不自动允许公开再分发清洗后的图片。课程交付可优先使用清洗清单、处理脚本和获准展示的示例。

### 第 5 页：Evaluation and project outcomes

建议改为 **Path to deployment**，将重复的模型训练指标收回第 2 页。

建议放三个阶段：

1. **Course prototype**：审计数据、训练基线、留出测试、离线 demo。能回答在现有数据上表现如何。
2. **Target-setting validation**：未来取得合规的目标设备照片和临床参考标签，检查跨设备、跨来源表现、关键类别漏检及工作量。需要合作和资源，目前尚未落实。
3. **Supervised pilot**：未来落实本地复核和转诊安排，在监督下评估可用性、转诊完成情况和运行问题。是否扩大使用取决于这些证据。

列出主要障碍与对应行动：专用设备到手机的图像差异需要目标场景验证；噪声标签需要可信参考标准；工具产生转诊建议之后需要真实接诊能力和维护责任。没有证据前，不声称已经临床可用。

课程可以提出指标和后续判定流程，但不要随意填“95% sensitivity”之类未经论证的门槛。具体临床阈值应结合用途、漏检后果和接诊容量制定。

## 现有备注与时间

仅计 ENGLISH SCRIPT 到中文解释之前的口语内容，不包含中文和 SOURCES AND ASSUMPTIONS：

| 页码 | 英文词数 | 120 词/分钟 | 130 词/分钟 |
|---|---:|---:|---:|
| 1 | 102 | 51 秒 | 47 秒 |
| 2 | 120 | 60 秒 | 55 秒 |
| 3 | 102 | 51 秒 | 47 秒 |
| 4 | 99 | 50 秒 | 46 秒 |
| 5 | 110 | 55 秒 | 51 秒 |
| 合计 | 533 | 约 4 分 27 秒 | 约 4 分 06 秒 |

这是基于字数的估算，不是实际排练结果。若只有 110 词/分钟，533 词已约 4 分 51 秒，几乎没有换人和停顿余量。

讲稿目前语法和逻辑基本正常，主要问题是第 2、5 页重复、反复出现“we will”，以及第 3 页未来功能的时态容易让人误解。备注中的双语解释和来源区可以保留，演讲时只朗读英文部分。

## 建议替换的英文讲稿

以下版本与上面的页面调整配套，保留学生项目的可执行范围，不代表项目已经完成。正式使用前要同步修改幻灯片，尤其是第 3、5 页。

### Slide 1

Eye Sentinel supports community health workers who need to assess eye problems where specialist care is difficult to reach. WHO estimates that at least one billion people have vision impairment that could have been prevented or remains unaddressed. That is the global context. Our project focuses on a narrower task: studying selected eye conditions visible in photographs, with priority given to potentially sight-threatening findings such as suspected corneal ulceration. Our first decision is which categories the available images can support. The course deliverable is a research prototype, with future screening and referral support as its intended application.

### Slide 2

Our starting data are EyeWorld and FrontofEyeDx. The project brief describes noisy web images, including photographs taken with specialist equipment. We will audit image types and labels, remove diagnostic text cues, and group duplicates and related panels before splitting the data. Final classes depend on usable sample counts and label quality. We propose pretrained MobileNetV3-Small as a lightweight baseline, using augmentation only during training and class weighting if needed. We will report per-class sensitivity, macro F1 and failure examples on held-out data. These results will measure performance on the available images. Performance on community phone photographs remains an open question.

### Slide 3

This slide shows the intended clinical workflow. A community health worker captures photographs and records symptoms. A future tool would check image usability, present findings and uncertainty, and prepare a case summary for review through the local referral pathway. For this course, we will implement the image classification component and an offline demonstration. The symptom questionnaire and referral interface describe the product direction. They do not imply that we have trained a multimodal model or validated referral decisions. Clinical use would require approved protocols, target-setting validation and a clear review process. The prototype does not establish diagnoses or recommend medication.

### Slide 4

Patients need privacy, understandable results and access to follow-up care. Community health workers need a short workflow and a clear route to request review. Receiving clinics need useful case information and enough capacity to handle referrals. A future operating program would also need responsibility for maintenance and data handling. During this course, our focus is responsible use of the existing images. We will follow the stated research or nonprofit restriction, check available usage information and avoid exposing identifying details. Where metadata permit, we will examine performance differences across sources or groups and state what we cannot assess.

### Slide 5

Our path has three stages. First, this course will produce an audited dataset inventory, a trained baseline and an offline demo, with documented errors and limitations. Second, future validation would require appropriately obtained photographs from the intended devices and setting, with reliable clinical reference labels. That would test whether performance transfers beyond the web datasets. Third, a supervised pilot would require clinical partners, a working referral pathway and clear operational responsibility. It would assess usability and whether referrals lead to care. The main barriers are image differences, label quality and service capacity. Each stage needs evidence before we move forward.

## 视觉调整顺序

1. 第 2 页：压缩每列文字，避免模型名与数据集名称之间显得拥挤，保持“macro F1”完整。
2. 第 3 页：用明确标注的概念输出或输入输出示例替换部分长句。
3. 第 5 页：用三个阶段呈现部署路线，并把各阶段的证据要求放在同一位置。
4. 如加入临床样本，先检查许可、去标识化及图像类型说明。不要把裂隙灯或眼底图误标为普通手机输入。
5. 不必增加装饰性眼睛图片，也不必把现有五页扩展成更长的主体。

## 建议准备的 appendix / Q&A

- 为什么不直接做 100 类？答：原始病种清单不等于可学习的高质量类别；先按图像类型、样本量和标签可信度确定范围。
- 为什么图像分类能帮助转诊？答：它是未来流程的一个组件；课程只验证分类，转诊价值需要独立的目标场景和流程证据。
- 怎么避免数据泄漏？答：对重复、近重复、多面板同源样本归组，尽可能按病例或患者划分，必要时开展跨来源测试。
- 为什么没有多模态模型？答：现有本地资料未证明存在配对图像和症状；缺乏配对数据时，不声称能训练和验证多模态模型。
- 没有眼科专家如何核实标签？答：课程能核查标签完整性和一致性，不能独立确认医学诊断；不确定样本进入复核队列，并报告这一局限。
- 如何证明能防止失明？答：课程离线分类结果不能证明健康结局改善；未来需要验证真实工作流和照护结果。

## 依据

- 本地旧稿：`../archive/proposal_versions/Eye_Sentinel_Project_Proposal_Student_Scope.pptx`，5 页及备注。
- 本地项目说明：../references/Project Opportunity_EyeSentinel.pdf，1 页。其推荐的多模态方案是项目方向，不是已经提供配对数据或必须在课程内完成全部临床功能的证据。
- 本地清洗手册：../references/Instructions to Derm-FrontofEyeDx Database.pdf，26 页。其操作指令作为当时的审阅材料参考；后续数据整理记录见 `../data/audit/`。
- WHO, Blindness and vision impairment，2026-02-10：https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment
- 用户截图：oral presentation rubric，总分 35 分，主要要求是 5 分钟，建议 5 页，允许 appendix；不要求所有组员都发言。

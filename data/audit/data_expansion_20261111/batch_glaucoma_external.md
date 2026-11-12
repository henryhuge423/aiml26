# 外部青光眼批次审计：RIM-ONE DL、BEH、G1020 部分镜像

审计日期：2026-11-11。范围仅为 `data/external_glaucoma/` 的 RIM-ONE DL、BEH 与 G1020 部分下载；没有移动、解压、训练或纳入任何图像。

## 结论与分流

| 来源 | 可用候选（尚不可直接纳入） | 已知重复 | 需要隔离 | 原因 |
|---|---:|---:|---:|---|
| RIM-ONE DL | 485 个唯一 payload（待跨库 pHash、许可用途复核） | 485 个第二份 split 副本 | 970 个 archive member 均隔离 | 同一 485 张图被随机/医院两种 split 各收录一次；两种 split 不可叠加。
| BEH | 573 个唯一且标签无精确冲突的 payload（待跨库 pHash、患者/双眼关联） | 47 个额外文件，属于 47 个精确重复组 | 28 个文件 / 14 个 exact-payload 组有 normal–glaucoma 标签冲突；全来源在许可和关联完成前仍隔离 | 冲突标签使这 14 个唯一 payload 不能作监督样本。
| G1020 镜像（部分） | 0（当前不构成随机或完整队列） | 0（当前本地 70 张内） | 70 张 | 第三方镜像、许可未由原始发布者核实、只取得 70/1020 张且下载次序非随机；`0/1` 语义未与原始发布核对。

“可用候选”仅指通过本地 ZIP/标签/精确文件审计后的条件性候选，**不代表准许训练或推理使用**。上述三来源保持隔离，直至完成许可、跨库近重复以及患者级关联审查。

## 包完整性与标签证据

| 文件 | SHA-256 | CRC | 图像/标签计数 |
|---|---|---|---|
| `rim_one_dl/images.zip` | `85aed5f95c794f52d11b6ed953032108a66576ffcff4b587365df467491b603d` | `zipfile.testzip()` 返回 `None`（971 全 member） | 970 PNG archive member；485 唯一 encoded payload。随机 split：train normal/glaucoma 219/120、test 94/52；医院 split：train 195/116、test 118/56。每一种 split 都是 normal 313、glaucoma 172。 |
| `beh/images.zip` | `948e36b00409ee2febd94f38f6906b9c918a8dca9da889524ae1c104133fbb27` | `zipfile.testzip()` 返回 `None`（1,275 全 member） | 634 图像：normal 463、glaucoma 171；另有 7 个 macOS/`.DS_Store` 非图像 member。 |
| `g1020_mirror/G1020.csv` | `9c8e0867a337261c5faeefeb1adead980f635cc41a27d692e60d7eb70dd770b6` | 不适用 | CSV 1,020 行：`binaryLabels` 0=724、1=296。现有 JPEG 70 张，均在 CSV 中：0=59、1=11。 |

RIM 以 encoded file SHA-256 分组：485 个组全都恰有两个文件（一个随机 split 路径和一个医院 split 路径），没有第三份或未配对 payload。因此任何“970 张 RIM”训练条目都是确定性重复/泄漏风险。

BEH 以 encoded file SHA-256 分组后只有 587 个唯一 payload：47 个重复组共产生 47 个额外文件。其中 22 组仅 normal、11 组仅 glaucoma，14 组横跨 normal 和 glaucoma 文件夹。这 14 组（28 文件）为已验证标签冲突，必须隔离；例如 `glaucoma/IM001336.JPG` 与 `normal/IM000294.JPG` 的 bytes 完全相同。

## 已解析元数据与患者/眼别风险

| 来源 | 可解析字段 | 患者/眼别结论 |
|---|---|---|
| RIM-ONE DL | split 类型、train/test、文件夹标签、文件名 | 未见显式 patient ID。485 个唯一文件名中 137 个含 `left` token、0 个含 `right` token；`r1/r2/r3` 前缀计数 98/250/137，不能作为可靠患者 ID 或配对规则。 |
| BEH | 文件夹诊断标签、文件名 | ZIP 内无患者 ID、眼别或采集日期。括号序号文件名不构成患者或左右眼字段。 |
| G1020 部分镜像 | `imageID`、`binaryLabels` | CSV 未提供患者 ID 或眼别；0/1 临床含义尚未从原始发布核对。 |

## 对 `formal_v3/manifest.csv` 的关联状态

正式清单有 29,240 行；其 source 名称中不含本批三个来源。对 RIM 485 个唯一 payload、BEH 587 个唯一 payload和 G1020 当前 70 个文件（共 1,189）计算 encoded-file SHA-256 并与该清单的 `sha256` 字段比对，得到 **0 个精确文件匹配**。这只排除字节完全相同的文件；完整清单的逐图 pHash 计算被中止，**本审计不能报告“无跨来源重复”**。

可复现的只读关联脚本为 `rim_beh_g1020_hash_audit.py`：它对 `formal_v3` 路径逐图构建 64-bit DCT pHash（灰度 32×32、8×8 DCT、去 DC 中位数阈值），并将海明距离 `<=4` 的候选输出为人工复核边，同时进行 encoded-file SHA-256 等值比对。首次执行没有进度 checkpoint，持续逐图解码超过数分钟，按审计协调指示中止；没有生成可据以宣称关联结果的 JSON。

后续应在带进度/可恢复 checkpoint 的运行中完成这一 pHash 步骤，并将命中与对应 `formal_v3` 样本的 `related_group` 合并审查；在那之前，本批数据不得写入训练或评估 manifest。

## 许可与来源边界

本结论沿用 `data/external_glaucoma/ACQUISITION.md` 中的获取记录：RIM-ONE DL 的作者限制研究/教育用途并要求维持比较 split；BEH 记录为 CC BY-NC 4.0 且要求引用；G1020 是未证实的第三方镜像。这里未重新裁决用途/商业合规性，故许可复核仍是纳入前置条件。

"""Build an editable five-slide revision with bilingual speaker notes.

Run with Python and python-pptx. The original deck remains unchanged.
"""
from pathlib import Path
import hashlib
import json
import re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'archive' / 'proposal_versions' / 'Eye_Sentinel_Project_Proposal_Student_Scope.pptx'
OUTPUT = ROOT / 'Eye_Sentinel_Project_Proposal_v3.pptx'
NAVY, TEAL, GRAY, MUTED = '162E46', '087F8C', '445768', '71808D'
FONT = 'Arial'
before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
prs = Presentation(SOURCE)
assert len(prs.slides) == 5
for slide in prs.slides:
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def text(slide, x, y, w, h, content, size=20, color=GRAY, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(content.split('\n')):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.space_after = Pt(7)
        para.line_spacing = 1.08
        run = para.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor.from_string(color)
    return box


def header(slide, number, title, subtitle=None):
    text(slide, .75, .53, 11.8, .64, title, 34, NAVY, True)
    if subtitle:
        text(slide, .75, 1.38, 11.8, .52, subtitle, 20, GRAY)
    text(slide, 12.25, 6.98, .38, .23, f'{number:02}', 11, MUTED)


scripts = ['Eye Sentinel asks how an eye-image model can make useful use of limited review time. Our intended setting is community eye care, where a health worker may need to share selected images for further review. We will build an offline prototype using the course photos. Our main question is simple: if only ten images out of a hundred can be selected, which ten should the model choose? We will compare ways to find incorrect predictions and images from categories the model has never learned. Four students will complete the data preparation, model training, experiments and local demo.', 'The course data contain 27,242 fundus photos, 4,886 Wide-Field photos and 2,321 FrontofEyeDx photos. These examples show different image types. We will start with three to five classes from Wide-Field, choosing photos with similar capture styles. Pterygium, blepharitis and hordeolum are candidates. Other selected categories will represent unfamiliar inputs. We will keep the categories used to develop our ranking separate from the unfamiliar categories used in the final test. Related images will stay together when we create training, validation and test groups. A pretrained MobileNetV3-Small will provide the image classifier.', 'We will compare three ways to rank images for review. The first uses the model score. The second checks whether predictions stay consistent after small changes in brightness, compression and scale. The third combines both signals. We will select five, ten or twenty images per hundred, then use the existing labels to count the errors and unfamiliar images selected. Each method will see the same test batches and receive the same review allowance. We will report errors left among the remaining images and results for each known class. We will also repeat the comparison with different proportions of unfamiliar images.', 'The demo will let a user open a batch of images, choose a review allowance and view the selected photos with their predictions. This makes the proposed workflow easy to inspect. Health workers need a short process and offline access. Patients benefit from clear wording and local image handling. Receiving clinics need the photo, model output and reason for selection together. As an extension, we will test whether extra predictions should be reserved for difficult images. We will compare one prediction for every image, repeated predictions for every image, and repeated predictions only for images selected by the first pass.', 'We will first prepare the image groups and class map, then train the classifier and compare the ranking methods. Finally, we will package a local demo and measure CPU time, memory use and model size on our laptop. Our main result will be a comparison of how many errors each strategy finds under the same review allowance. The computing extension will show what extra processing costs and what it adds. We will address varied photos by selecting a consistent image type, label conflicts through a saved review list, and limited computing with a small model. Each of our four members will lead one part of the work.']
translations = ['Eye Sentinel 研究怎样把有限的图片复核时间用在更值得查看的图片上。设想的场景是社区眼健康服务：卫生工作者需要从一批图片中选出部分图片，供进一步查看。本组使用课程照片制作离线原型。核心问题很简单：如果每一百张图片只能选出十张，模型应该选哪十张？我们比较不同选图方法，看它们能找出多少错误预测和模型没有学过的类别。四人完成数据整理、模型训练、实验和本地演示。', '课程数据包括 27,242 张眼底照、4,886 张 Wide-Field 图片和 2,321 张 FrontofEyeDx 图片。这些样本展示了不同拍摄类型。我们先从 Wide-Field 选择三到五个拍摄方式接近的类别，候选包括翼状胬肉、睑缘炎和麦粒肿。另选其他类别作为模型没有学过的输入；用于开发选图方法的类别和最终测试类别分开。划分训练、验证和测试集时，相关图片放在同一组。分类器采用预训练 MobileNetV3-Small。', '我们比较三种复核排序方法：看模型分数；看轻微改变亮度、压缩和尺寸后预测是否一致；结合两种信号。分别设置每一百张图片选五张、十张或二十张，再用已有标签统计选出了多少错误预测和未知类别图片。各方法使用相同的测试批次和额度。同时报告剩余图片中的错误，以及每个已知类别的表现，并改变测试批次中未知类别的比例，重复比较。', '演示中可以打开一批图片，设置复核额度，再查看选中的图片及预测。这能直观展示设想流程。卫生工作者需要步骤简单、离线可用；患者需要用语清楚、图片在本地处理；接诊诊所需要一起看到照片、模型输出和选图原因。扩展实验研究额外计算该用在哪些图片上：比较全部只预测一次、全部重复预测，以及只对第一轮选出的难图重复预测。', '先整理图片分组和类别对应表，再训练分类器、比较选图方法，最后打包本地演示，在笔记本上测 CPU 时间、内存和模型大小。主要结果是各方法在相同复核额度下找出的错误数量。计算扩展实验展示额外处理花多少时间、带来什么变化。针对照片差异，选用接近的拍摄类型；标签冲突单独记录；计算资源有限则使用小模型。四位组员各负责一部分工作。']
source_notes = ['Project brief: Project Opportunity_EyeSentinel.pdf. Global-health motivation: WHO digital health guideline discusses connectivity, electricity and usability, https://www.ncbi.nlm.nih.gov/books/NBK541898/ ; WHO integrated eye care, https://www.who.int/activities/integrating-eye-care-in-health-systems . Review allowances are simulated experimental settings: 5, 10 or 20 images per 100. The course evaluates image selection using existing labels.', 'Audit dated 2026-10-22: Fundoscopy 27,242; Wide-Field 4,886; FrontofEyeDx 2,321; separate conflict review 22. Candidate GoogleBing-InfectEye counts: Pterygium 153, Blepharitis 99, Hordeolum 121. Select 3–5 known classes after checking capture style and label granularity. Assign all images of any unfamiliar class to either development or final testing; merge synonymous or overlapping labels in the class map before assignment. Hold exact/near duplicates and related panels in one group across the full selected pool. Use available case IDs. Save known train/validation/test groups and unfamiliar development/test classes. Cleaning is part of preparation.', 'Baseline classifier: pretrained MobileNetV3-Small. Ranking signals: uncertainty from maximum softmax; disagreement among predictive distributions under mild brightness, JPEG compression and scale changes; validation-weighted combination. Use identity plus three mild views, preserving the full image. Choose transform strengths, signal normalization and combination weight using known validation plus unfamiliar development classes. Freeze all settings before final testing. Define flaggable errors as wrong predictions on known classes plus any unfamiliar image assigned a known label. For each identical 100-image batch, select the top 5/10/20 by rank. Use unfamiliar shares of 0%, 10%, 30% as simulation settings and report each scenario separately. Repeat seeded group-respecting draws; report sample counts, spread, error capture, unfamiliar capture, unselected known-class error and per-class selection. Prior work: SelectiveNet, https://proceedings.mlr.press/v97/geifman19a.html ; test-time confidence estimation, https://arxiv.org/abs/2006.16705 . Contribution: empirical image-selection comparison under review and compute constraints.', 'Interface mockup and course demo: existing local images; batch upload, simulated allowance, ranked photos, predicted class, model score and selection reason. Stakeholders are design considerations from the brief. Extension: compare single-pass, four-view processing for all images, and four-view processing only for high-uncertainty images. Tune first-pass gate on validation data. Compare error capture at matched CPU-time budgets and report actual total time, p95 batch time, peak process memory and model file size. Local CPU measurements use the same input size and hardware, warm-up, fixed thread count and repeated runs. Scores and rankings remain model outputs.', 'Deliverables: code, class map, group IDs and fixed split files, trained classifier, review-budget curves and local demo. Main analysis remains ranking at equal review allowances; extension measures compute allocation. Four roles: data, model, evaluation, demo/report; cross-check splits and results. Global-health fit: limited review capacity and local processing become experimental constraints. Main obstacles: capture-style variation, folder-label conflicts and compute limits. Measure CPU performance on the available laptop and verify the demo with networking disconnected.']

from pptx.enum.shapes import MSO_SHAPE
import csv
from PIL import Image

def panel(slide,x,y,w,h,color='EFF4F6'):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(color); sh.line.fill.background()
    return sh

def photo(slide,path,x,y,w,h):
    with Image.open(path) as im: iw,ih=im.size
    scale=min(w/iw,h/ih); pw,ph=iw*scale,ih*scale
    slide.shapes.add_picture(str(path),Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))

rows=list(csv.DictReader((ROOT/'data/audit/current/image_files.csv').open(encoding='utf-8-sig')))
def sample(suffix):
    row=next(r for r in rows if r['path'].endswith(suffix))
    return ROOT/'data'/row['path']
fundus=sample('ACRIMA/Glaucoma/Im310_g_ACRIMA.jpg')
wide=sample('GoogleBing-InfectEye/Pterygium/Pterygium 002535.jpg')
front=sample('frontofeyedx/Dataset/Pterygium/image_20251119_134417_Pterygium.jpg')


s=prs.slides[0]
text(s,.75,.65,11.8,.9,'Eye Sentinel',54,NAVY,True)
text(s,.78,1.82,11.6,.7,'Making limited image-review time count',29,NAVY)
text(s,.78,2.95,7.6,1.0,'Community eye care needs a simple way\nto select photos for further review.',22)
text(s,.78,4.35,7.7,1.15,'Only 10 out of 100 can be reviewed.\nWhich images should we choose?',27,TEAL,True)
panel(s,9.0,2.95,3.5,2.8)
text(s,9.25,3.25,3.0,.5,'Our course project',22,NAVY,True)
text(s,9.25,4.05,3.0,1.35,'4 students\nExisting eye photos\nAn offline prototype',20)
text(s,.78,6.05,11.7,.55,'Compare image-selection strategies under a simulated review allowance.',19,NAVY)
text(s,.78,6.8,11.6,.35,'Xinhao Jin   ·   Zihan Fan   ·   Zheng Jian   ·   Barzin Badiee',15,NAVY)
text(s,12.25,6.98,.38,.23,'01',11,MUTED)

s=prs.slides[1]
header(s,2,'Existing photos, clearly separated test groups','Train on 3–5 Wide-Field classes; reserve other categories as unfamiliar inputs')
for x,title,count,path,label in [
    (.75,'EyeWorld / Fundoscopy','27,242 photos',fundus,'Folder label: Glaucoma'),
    (4.93,'EyeWorld / Wide-Field','4,886 photos',wide,'Folder label: Pterygium'),
    (9.1,'FrontofEyeDx','2,321 photos · 94 folders',front,'Folder label: Pterygium')]:
    panel(s,x,2.18,3.5,3.15)
    text(s,x+.15,2.36,3.2,.5,title,18,NAVY,True)
    photo(s,path,x+.2,3.02,3.1,1.25)
    text(s,x+.15,4.46,3.2,.42,count,19,TEAL,True)
    text(s,x+.15,4.98,3.2,.25,label,11,MUTED)
text(s,.75,5.67,11.8,.55,'Group related images → train MobileNetV3-Small → rank test images',21,NAVY,True)
text(s,.75,6.4,11.8,.5,'Use separate unfamiliar categories for method development and final testing.',18)

s=prs.slides[2]
header(s,3,'Which strategy finds more errors for review?','Give each method the same images and the same allowance')
for x,title,body in [(.75,'Model score','Rank images by how\nuncertain the model is'),(4.93,'View agreement','Check agreement after\nsmall image changes'),(9.1,'Combined signals','Use both the score\nand the agreement')]:
    panel(s,x,2.22,3.5,1.8)
    text(s,x+.16,2.43,3.18,.65,title,21,TEAL,True)
    text(s,x+.16,3.23,3.18,.65,body,17)
panel(s,.75,4.38,11.85,.83)
text(s,.98,4.58,11.35,.5,'Select 5, 10 or 20 images per 100  →  check selections against existing labels',20,NAVY,True)
text(s,.75,5.63,11.8,.55,'Count errors found, unfamiliar images found, and errors left in the rest.',20,NAVY)
text(s,.75,6.4,11.8,.42,'Repeat with 0%, 10% and 30% unfamiliar images; report each known class.',17)

s=prs.slides[3]
header(s,4,'A local review queue, with a choice of allowance','Open a batch → choose the allowance → view selected photos and predictions')
for y,title,body in [(2.18,'Health workers','Few steps and offline access'),(3.4,'Patients','Clear wording and local image handling'),(4.62,'Receiving clinics','Photo, output and selection reason together')]:
    text(s,.75,y,5.85,.42,title,22,TEAL,True)
    text(s,.75,y+.55,5.85,.55,body,18)
panel(s,7.28,2.15,5.25,3.65)
text(s,7.55,2.37,4.7,.4,'Proposed demo screen',21,NAVY,True)
photo(s,wide,7.55,3.02,1.9,1.35)
text(s,9.7,3.02,2.55,1.35,'Review allowance\nSelected image\nReason for selection',16,TEAL,True)
text(s,7.55,4.75,4.7,.4,'Photo  |  Predicted class  |  Score',18,NAVY)
text(s,7.55,5.33,4.7,.25,'Existing dataset image · planned interface',11,MUTED)
text(s,.75,6.08,11.8,.4,'Extension: use extra predictions only for images flagged by the first pass.',18,NAVY,True)
text(s,.75,6.63,11.8,.32,'Compare one pass, repeated passes for all, and repeated passes for selected images.',16)

s=prs.slides[4]
header(s,5,'Build, compare, and run offline','Main result: errors found at equal review allowances; extension: computing cost')
for x,title,body in [(.75,'01  Prepare','Choose classes; clean photos\nSave groups and test lists'),(4.93,'02  Compare','Train one classifier\nCompare ranking methods'),(9.1,'03 Package','Build the review queue\nMeasure time, RAM and MB')]:
    panel(s,x,2.18,3.5,1.75)
    text(s,x+.16,2.4,3.18,.5,title,23,TEAL,True)
    text(s,x+.16,3.05,3.18,.7,body,16)
for y,challenge,response in [(4.35,'Varied photos','Choose a consistent capture style; group related images'),(5.0,'Label conflicts','Keep the class map and a separate review list'),(5.65,'Limited computing','Use a small model; measure performance on our CPU')]:
    text(s,.75,y,3.1,.4,challenge,19,NAVY,True)
    text(s,4.1,y,8.4,.45,response,17)
text(s,.75,6.6,11.8,.36,'Four roles: data   ·   model   ·   evaluation   ·   demo and report',17,TEAL,True)

counts = []
for i, slide in enumerate(prs.slides):
    count = len(scripts[i].split())
    counts.append(count)
    slide.notes_slide.notes_text_frame.text = (
        'ENGLISH SCRIPT\n' + scripts[i] + '\n\n中文理解稿\n' + translations[i]
        + '\n\nSOURCES AND PROJECT DETAILS\n' + source_notes[i]
    )
prs.core_properties.title = 'Eye Sentinel: project proposal v3'
prs.core_properties.subject = 'Five-minute course presentation with bilingual speaker notes'
prs.core_properties.author = 'Xinhao Jin; Zihan Fan; Zheng Jian; Barzin Badiee'
prs.save(OUTPUT)
assert before == hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'Original deck changed'

# Verify saved content and geometry, rather than trusting the in-memory build.
saved = Presentation(OUTPUT)
assert len(saved.slides) == 5
issues = []
for i, slide in enumerate(saved.slides, 1):
    for shape in slide.shapes:
        if shape.left < 0 or shape.top < 0 or shape.left+shape.width > saved.slide_width or shape.top+shape.height > saved.slide_height:
            issues.append(f'Slide {i}: shape outside canvas')
    notes = slide.notes_slide.notes_text_frame.text
    assert all(k in notes for k in ['ENGLISH SCRIPT', '中文理解稿', 'SOURCES AND PROJECT DETAILS'])
    assert scripts[i-1] in notes
assert not issues, issues
audit = {
    'slides': len(saved.slides), 'words_per_slide': counts,
    'total_spoken_words': sum(counts),
    'spoken_seconds_at_120_wpm': round(sum(counts)/2),
    'original_sha256': before, 'original_unchanged': True,
    'bounds_issues': issues, 'visual_review': 'Required after PowerPoint rendering',
}
report_path = ROOT / 'tools' / 'build_reports' / 'Eye_Sentinel_v3_build_check.json'
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
print(OUTPUT)

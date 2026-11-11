"""Build the five-slide v4 proposal and the dataset PDF from its Markdown source."""
from pathlib import Path
import csv
import hashlib
import json
import re
from xml.sax.saxutils import escape
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Eye_Sentinel_Project_Proposal_v4.pptx'
NAVY, TEAL, GRAY, MUTED = '162E46', '087F8C', '445768', '71808D'
PALE, GOLD, GREEN, RED = 'EFF4F6', 'A66B0A', '28734E', 'A43B40'

def text(s, x, y, w, h, content, size=20, color=GRAY, bold=False):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(content.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        p.line_spacing = 1.05
        r = p.add_run(); r.text = line
        r.font.name = 'Arial'; r.font.size = Pt(size)
        r.font.bold = bold; r.font.color.rgb = RGBColor.from_string(color)
    return box

def panel(s, x, y, w, h, color=PALE):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(color)
    sh.line.fill.background()
    return sh

def photo(s, path, x, y, w, h):
    with Image.open(path) as im: iw, ih = im.size
    sc = min(w/iw, h/ih)
    s.shapes.add_picture(str(path), Inches(x+(w-iw*sc)/2), Inches(y+(h-ih*sc)/2), width=Inches(iw*sc), height=Inches(ih*sc))

def header(s, n, title, subtitle):
    text(s, .75, .53, 11.85, .65, title, 33, NAVY, True)
    text(s, .75, 1.38, 11.85, .65, subtitle, 19)
    text(s, 12.25, 6.98, .38, .23, f'{n:02}', 11, MUTED)

SCRIPTS = [
"Our project, Eye Sentinel, focuses on glaucoma screening in community clinics. A health worker takes a photo of the retina, the back of the eye. If the photo is blurred or too dark, signs of disease may be hard to see. The model may still give a confident answer. We want the system to check the photo first and help the health worker take a better one. It then uses image quality and model uncertainty to choose difficult cases for expert review. We will test whether this approach catches more missed glaucoma cases when experts can review ten out of every hundred images.",
"We have about 7,700 fundus images, split between glaucoma and normal eyes. The table shows the exact counts for each source. We will set aside the 705 ACRIMA images for the final test. This leaves about 7,000 images for model development and internal testing. Before splitting them, we will group each original photo with its edited or augmented versions. ODIR adds a separate set of 3,500 patients with photos of both eyes. We will keep each patient's two eyes in the same split. These groups let us compare methods on the same test images and trace every image back to its source.",
"Here is how the system works. First, it checks whether the photo is sharp, well exposed and clear enough to assess. If it needs a retake, the screen gives a reason, such as poor focus or low brightness. Next, a small model called MobileNetV3 estimates the probability of glaucoma. We calibrate these probabilities using a separate set of images. The system then combines image quality and model uncertainty to choose cases for expert review. Clear photos with a low glaucoma probability receive a negative screening result. Positive screens and photos that remain unreadable receive a referral record. We count the total number of patients needing follow-up.",
"We will compare three ways to choose images for review: random selection, model uncertainty, and our combined score. Each method uses the same model and can select ten out of every hundred images. We will also test five and twenty. The tests include original photos and versions with blur, brightness changes or lower contrast. The main question is how many glaucoma cases receive a negative screening result. We also ask how many negative results are correct. Our goal is more than 98 out of every hundred. Alongside this, we will report how many people receive negative results, how many errors expert review catches, and how many patients need a referral.",
"The demo runs offline on a laptop. A health worker imports a batch of photos and sees three lists. Green shows negative screening results. Yellow asks for a retake and explains the image problem. Red shows positive screens, cases selected for expert review, and photos that remain unreadable. Each referral record includes the original photo, the glaucoma probability and the reason for referral. Our target is to process each image in under two hundred milliseconds, using less than five hundred megabytes of memory. The final submission will include the data lists, code, experiment results and demo. Our four team members will share the data, model, evaluation and demo work."
]
CHINESE = [
'我们的项目叫 Eye Sentinel，关注社区里的青光眼筛查。基层工作人员拍摄眼底照片时，如果画面模糊或偏暗，病变就可能看不清，模型却仍然给出很有把握的判断。我们希望系统先检查照片，帮助工作人员把图拍清楚，再结合图像质量和模型的不确定度，把难例交给专家。我们要回答的问题是：每一百张图只能复核十张时，这种方法能多找出多少原本会漏掉的青光眼？',
'现有数据约有 7,700 张眼底图，包括青光眼和正常眼底，表格列出了各来源的具体数量。我们会留出 ACRIMA 的 705 张图做最终测试，剩余约 7,000 张用于模型开发和内部测试。划分前，先把原图及其编辑、增强版本放在同一组。ODIR 还提供了 3,500 位患者的双眼照片，同一患者的两只眼也会放进同一个集合。各方法共用这些分组和测试图，每张图都保留来源记录。',
'系统先检查照片是否清楚、曝光是否合适。需要重拍时，屏幕会说明原因，比如对焦不清或画面太暗。接着，MobileNetV3 估计青光眼概率，我们用单独的一组图像校准这些概率。系统再结合图像质量和模型的不确定度，选出需要专家复核的图片。质量合格、患病概率低且未入选复核的图片给出筛查阴性结果；筛查阳性和重拍后仍看不清的图片生成转诊记录。最后统计有多少患者需要后续处理。',
'我们比较三种选图方式：随机选择、按模型不确定度选择，以及两种信号结合。各方法使用同一个模型，每一百张图选十张复核，也会测试选五张和二十张的情况。测试包括原图，以及加入模糊、亮度变化和低对比度的版本。最关心的是有多少青光眼被判为筛查阴性，也会看阴性结果中有多少是正确的，目标是每一百个超过九十八个。同时报告阴性结果覆盖多少人、复核找出了多少错误，以及多少患者需要转诊。',
'演示可以在笔记本上离线运行。工作人员导入一批照片后，会看到三个列表：绿色显示筛查阴性，黄色提示重拍并说明原因，红色显示筛查阳性、需要专家复核的病例，以及重拍后仍看不清的照片。每份转诊记录附上原图、青光眼概率和转诊原因。我们希望单张图处理时间低于 200 毫秒，内存使用低于 500 MB。最终提交数据清单、代码、实验结果和演示程序，四位成员分工完成数据、模型、评估和演示。'
]
NOTES = [
'Research question: compare missed glaucoma cases at a fixed expert-review budget using quality and uncertainty together. Main budget: 10 images per 100. Setting: community fundus screening. Calibration reference: https://proceedings.mlr.press/v70/guo17a.html .',
'Local audit 2026-10-22, source_class_counts.csv: EyeAugmented 2325 G + 2615 N; Kaggle 1005 G + 1074 N; ACRIMA 396 G + 309 N. Total 7724, development 7019. ACRIMA original study: https://pmc.ncbi.nlm.nih.gov/articles/PMC6425593/ . ACRIMA and cross-source related versions stay outside development. ODIR 7000 images / 3500 orig_id is a separate extension. Patient grouping applies where IDs exist. Other sources require image-family grouping, including augmentation parents. Group split target 70/15/15, validation subdivided into calibration and policy development. Final sizes follow grouping.',
'Q in [0,1] from normalized Laplacian variance, RMS contrast and exposure fractions within retinal mask, standardized image size. U = -[p ln(p)+(1-p)ln(1-p)]/ln(2). S = wU+(1-w)(1-Q). Tune normalization, w, quality threshold and positive threshold t on development data only. Score is review priority. Negative requires adequate quality, p<t and no review selection. Persistent ungradable images receive referral. Track positive referral union review union ungradable cases by patient. If the combined workload exceeds a capacity scenario, report the overflow. A single quality score describes the image; improvement after a retake indicates correctability.',
'Same model, retake gate, probability threshold and calibration across ranking strategies. Quality-only ablation added. Review budget denominator: images entering evaluation after retake handling. Full workflow workload denominator: all initial inputs, grouped by patient when available. Miss rate = G in released-negative / all G. NPV = true normal in released-negative / all released-negative. Target NPV >98%, reported with observed prevalence, counts and 95% group-bootstrap CI. Reweight prevalence scenarios 2%,5%,10%. Review recall = G reviewed/all G; precision=G reviewed/all reviewed; error capture = original classification errors reviewed/all original classification errors. Retained AUC, sensitivity, specificity and class counts. Blur sigma .5/1/2 at 224 px, brightness .6/.8/1.2/1.5, contrast .8/.6/.4. Pair source and degraded versions; simulated retakes restore the source image, an ideal restoration scenario. Selective prediction reference: https://proceedings.mlr.press/v97/geifman19a.html .',
'Proposed interface: yellow retake prompts describe observed blur/exposure; persistent ungradable moves to red. Red includes positive screens and selected review cases. Green includes only negative screening results meeting the decision rule. Export full image and verified optic-disc crop where available, probability and quality reason. CPU targets are engineering goals measured after warm-up at fixed input size/thread count: median and p95 inference, peak RSS, batch time and model size. Capture and retake time measured separately. Deliverables include source/family/patient split manifests, IQA, calibration and scoring code, budget curves, source-held-out evaluation and offline demo.'
]

def build_deck():
    source = ROOT / 'Eye_Sentinel_Project_Proposal_v3.pptx'
    if not source.exists():
        source = ROOT / 'archive/proposal_versions/Eye_Sentinel_Project_Proposal_v3.pptx'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    prs = Presentation(source)
    for s in prs.slides:
        for sh in list(s.shapes): sh._element.getparent().remove(sh._element)
    rows = list(csv.DictReader((ROOT/'data/audit/current/image_files.csv').open(encoding='utf-8-sig')))
    fundus = ROOT/'data'/next(r['path'] for r in rows if r['path'].endswith('ACRIMA/Glaucoma/Im310_g_ACRIMA.jpg'))
    s = prs.slides[0]
    text(s,.75,.6,11.8,.85,'Eye Sentinel',52,NAVY,True)
    text(s,.78,1.65,11.8,1.1,'Quality-Aware Selective Screening and Referral\nfor Community Glaucoma Care',27,NAVY)
    text(s,.78,3.05,7.25,1.1,'A blurred fundus photo can still produce\na confident prediction.',26,NAVY,True)
    text(s,.78,4.52,7.4,1.25,'Check image quality, guide a retake,\nand prioritize difficult cases for review.',23,TEAL)
    photo(s,fundus,9.0,2.95,3.35,2.75)
    text(s,9.02,5.85,3.4,.55,'ACRIMA fundus image\nFolder label: Glaucoma',12,MUTED)
    text(s,.78,6.38,11.6,.3,'Offline prototype with a 10% expert-review allowance',18,NAVY,True)
    text(s,.78,6.94,11.5,.28,'Xinhao Jin   ·   Zihan Fan   ·   Zheng Jian   ·   Barzin Badiee',14,NAVY)
    text(s,12.25,6.98,.38,.23,'01',11,MUTED)

    s = prs.slides[1]
    header(s,2,'Fundus images and data splits','7,724 images for glaucoma classification')
    data=[['Source','Glaucoma','Normal','Use'],['EyeAugmented','2,325','2,615','Development'],['Kaggle eye-disease subset','1,005','1,074','Development'],['ACRIMA','396','309','Held-out source']]
    tab=s.shapes.add_table(4,4,Inches(.75),Inches(2.3),Inches(11.85),Inches(2.05)).table
    for col,w in zip(tab.columns,[5.0,1.65,1.65,3.55]): col.width=Inches(w)
    for i,row in enumerate(data):
        for j,value in enumerate(row):
            c=tab.cell(i,j); c.text=value
            c.fill.solid(); c.fill.fore_color.rgb=RGBColor.from_string(NAVY if i==0 else ('EFF4F6' if i%2 else 'FFFFFF'))
            c.margin_left=Inches(.13); c.margin_top=Inches(.11)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name='Arial';r.font.size=Pt(18);r.font.bold=i==0
                    r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else NAVY)
    text(s,.75,4.78,5.65,.5,'7,019 images for development',24,TEAL,True)
    text(s,.75,5.43,5.65,.95,'Group original images with their\naugmented and cropped versions.',19)
    text(s,7.05,4.78,5.5,.5,'3,500 ODIR patient pairs',24,TEAL,True)
    text(s,7.05,5.43,5.5,.95,'Keep each patient’s two eyes\nin the same split using orig_id.',19)
    text(s,.75,6.62,11.8,.35,'Final test: all 705 ACRIMA images, with their related versions grouped together.',17,NAVY)

    s=prs.slides[2]
    header(s,3,'Screening and referral workflow','Quality checks guide retakes. The model estimates glaucoma probability.')
    for x,title,body in [(.75,'1  Image quality','Sharpness, contrast, exposure\nRetake prompt with a reason'),(4.93,'2  Glaucoma model','MobileNetV3-Small\nCalibrated probabilities'),(9.1,'3  Expert review','Uncertainty + image quality\nSelect cases within the budget')]:
        panel(s,x,2.25,3.5,1.7)
        text(s,x+.17,2.47,3.16,.4,title,21,TEAL,True)
        text(s,x+.17,3.12,3.16,.65,body,16)
    text(s,.75,4.32,11.8,.6,'Review score = w × uncertainty + (1 − w) × (1 − quality)',24,NAVY,True)
    for x,c,title,body in [(.75,GOLD,'Retake','Poor image quality\nCheck the new photo'),(4.93,GREEN,'Screen negative','Clear photo, low probability\nOutside the review queue'),(9.1,RED,'Refer / review','Positive screen, selected case\nor photo still unreadable')]:
        text(s,x,5.32,3.5,.4,title,22,c,True)
        text(s,x,5.91,3.5,.7,body,16)
    text(s,.75,6.72,11.8,.34,'Count each patient once when reporting the total referral workload.',16,NAVY)

    s=prs.slides[3]
    header(s,4,'Comparing review methods','Main outcome: glaucoma cases that receive a negative screening result')
    for x,title,body in [(.75,'Random review','Repeated random selection'),(4.93,'Standard AI','Calibrated uncertainty'),(9.1,'Quality-aware AI','Uncertainty + image quality')]:
        panel(s,x,2.22,3.5,1.12)
        text(s,x+.16,2.4,3.18,.38,title,22,TEAL,True)
        text(s,x+.16,2.95,3.18,.28,body,15)
    text(s,.75,3.78,5.7,.5,'10% main review budget',27,NAVY,True)
    text(s,.75,4.48,5.7,.95,'Also compare 5% and 20%\nTest original and degraded photos',18)
    text(s,.75,5.8,5.7,.75,'Blur, brightness and contrast changes\nFinal test on ACRIMA',18)
    text(s,7.05,3.78,5.5,.5,'Missed glaucoma cases',25,NAVY,True)
    text(s,7.05,4.48,5.5,1.3,'NPV and share of negative results\nErrors caught by expert review\nSensitivity and total referrals',19)
    text(s,7.05,6.12,5.5,.5,'Goal: NPV >98% in the negative group',17,TEAL,True)
    text(s,.75,6.72,11.8,.34,'Same classifier, retake rules and screening threshold across ranking methods.',16,NAVY)

    s=prs.slides[4]
    header(s,5,'Offline demo and final submission','Import photos, follow retake prompts and view screening results')
    for x,c,title,body in [(.75,GREEN,'Green: screen negative','Screening result\nRoutine follow-up prompt'),(4.93,GOLD,'Yellow: retake','Reason for the retake\nImport the new photo'),(9.1,RED,'Red: refer / review','Probability and referral reason\nOriginal photo and record')]:
        panel(s,x,2.2,3.5,1.82)
        text(s,x+.16,2.43,3.18,.5,title,20,c,True)
        text(s,x+.16,3.2,3.18,.65,body,16)
    text(s,.75,4.55,3.6,.5,'Data and code',23,NAVY,True)
    text(s,.75,5.22,3.6,1.0,'Image groups and data splits\nQuality checks and model code',18)
    text(s,4.93,4.55,3.6,.5,'Experimental report',23,NAVY,True)
    text(s,4.93,5.22,3.6,1.0,'Missed cases at each budget\nResults by source and quality',18)
    text(s,9.1,4.55,3.5,.5,'CPU targets',23,NAVY,True)
    text(s,9.1,5.22,3.5,1.0,'<200 ms per image\n<500 MB peak memory',21,TEAL,True)
    text(s,.75,6.65,11.8,.4,'Four roles: data preparation, modeling, evaluation, demo and report',17,NAVY)
    for i,s in enumerate(prs.slides):
        s.notes_slide.notes_text_frame.text='ENGLISH SCRIPT\n'+SCRIPTS[i]+'\n\n中文理解稿\n'+CHINESE[i]+'\n\nSOURCES AND PROJECT DETAILS\n'+NOTES[i]
    prs.core_properties.title='Eye Sentinel: Quality-Aware Selective Screening and Referral, v4'
    prs.core_properties.subject='Five-minute proposal with bilingual speaker notes'
    prs.core_properties.author='Xinhao Jin; Zihan Fan; Zheng Jian; Barzin Badiee'
    prs.save(OUT)
    saved=Presentation(OUT)
    assert len(saved.slides)==5
    bounds=[]
    for n,s in enumerate(saved.slides,1):
        for sh in s.shapes:
            if sh.left<0 or sh.top<0 or sh.left+sh.width>saved.slide_width or sh.top+sh.height>saved.slide_height: bounds.append(n)
        assert CHINESE[n-1] in s.notes_slide.notes_text_frame.text
    assert not bounds, bounds
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    return {'slides':5,'script_words':[len(x.split()) for x in SCRIPTS], 'source_v3_unchanged':True,'bounds_issues':bounds}

def build_pdf():
    pdfmetrics.registerFont(TTFont('YaHei',r'C:\Windows\Fonts\msyh.ttc'))
    pdfmetrics.registerFont(TTFont('YaHei-Bold',r'C:\Windows\Fonts\msyhbd.ttc'))
    pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHei-Bold',italic='YaHei',boldItalic='YaHei-Bold')
    styles={
        'body':ParagraphStyle('body',fontName='YaHei',fontSize=11,leading=18.5,spaceAfter=10,wordWrap='CJK',textColor=colors.HexColor('#203040')),
        'title':ParagraphStyle('title',fontName='YaHei-Bold',fontSize=23,leading=32,spaceAfter=17,textColor=colors.HexColor('#162E46')),
        'h2':ParagraphStyle('h2',fontName='YaHei-Bold',fontSize=17,leading=25,spaceAfter=15,textColor=colors.HexColor('#087F8C')),
        'h3':ParagraphStyle('h3',fontName='YaHei-Bold',fontSize=12,leading=20,spaceBefore=10,spaceAfter=8),
        'cell':ParagraphStyle('cell',fontName='YaHei',fontSize=9.5,leading=15.5,wordWrap='CJK'),
        'th':ParagraphStyle('th',fontName='YaHei-Bold',fontSize=9.5,leading=15.5,textColor=colors.white,wordWrap='CJK')}
    def markup(t):
        t=escape(t)
        t=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#087F8C">\1</link>',t)
        t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
        return t.replace('`','')
    lines=(ROOT/'Eye_Sentinel_Dataset_Introduction.md').read_text(encoding='utf-8-sig').splitlines()
    story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line=='<!-- pagebreak -->':story.append(PageBreak());i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in row): rows.append(row)
                i+=1
            widths=([156,64,72,195] if len(rows[0])==4 else [135,352])
            cells=[[Paragraph(markup(t),styles['th' if n==0 else 'cell']) for t in row] for n,row in enumerate(rows)]
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#162E46')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EFF4F6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
            story.extend([table,Spacer(1,12)]);continue
        if line.startswith('# '):style='title';line=line[2:]
        elif line.startswith('## '):style='h2';line=line[3:]
        elif line.startswith('### '):style='h3';line=line[4:]
        else:style='body';line=line.removeprefix('- ')
        story.append(Paragraph(markup(line),styles[style]));i+=1
    def footer(c,d):
        c.setFont('YaHei',8);c.setFillColor(colors.HexColor('#71808D'))
        c.drawString(54,28,'Eye Sentinel   /   数据与实验方案   /   2026-10-23')
        c.drawRightString(A4[0]-54,28,str(d.page))
    path=ROOT/'Eye_Sentinel_Dataset_Introduction.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=45,bottomMargin=48,title='Eye Sentinel 数据集与实验方案',author='Eye Sentinel Team')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    reader=PdfReader(path)
    assert len(reader.pages)==6, f'Expected six pages, got {len(reader.pages)}'
    return {'dataset_pages':len(reader.pages)}

if __name__=='__main__':
    report=build_deck();report.update(build_pdf())
    headings=['Community glaucoma screening','Fundus images and data splits','Screening and referral workflow','Comparing review methods','Offline demo and final submission']
    script=['# Eye Sentinel v4 演讲稿', '', '五页英文讲稿及对应中文理解稿。英文全文约 '+str(sum(len(s.split()) for s in SCRIPTS))+' 词，按每分钟 120 词讲述约需 '+str(round(sum(len(s.split()) for s in SCRIPTS)/120,1))+' 分钟。', '']
    for n,(title,en,zh) in enumerate(zip(headings,SCRIPTS,CHINESE),1):
        script.extend([f'## Slide {n}: {title}', '', '### English', '', en, '', '### 中文理解稿', '', zh, ''])
    (ROOT/'Eye_Sentinel_Speaker_Notes_v4.md').write_text('\n'.join(script),encoding='utf-8')
    path=ROOT/'tools/build_reports/Eye_Sentinel_v4_build_check.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

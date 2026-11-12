# Eye Sentinel v4 演讲稿

五页英文讲稿及对应中文理解稿。英文全文约 530 词，按每分钟 120 词讲述约需 4.4 分钟。

## Slide 1: Community glaucoma screening

### English

Our project, Eye Sentinel, focuses on glaucoma screening in community clinics. A health worker takes a photo of the retina, the back of the eye. If the photo is blurred or too dark, signs of disease may be hard to see. The model may still give a confident answer. We want the system to check the photo first and help the health worker take a better one. It then uses image quality and model uncertainty to choose difficult cases for expert review. We will test whether this approach catches more missed glaucoma cases when experts can review ten out of every hundred images.

### 中文理解稿

我们的项目叫 Eye Sentinel，关注社区里的青光眼筛查。基层工作人员拍摄眼底照片时，如果画面模糊或偏暗，病变就可能看不清，模型却仍然给出很有把握的判断。我们希望系统先检查照片，帮助工作人员把图拍清楚，再结合图像质量和模型的不确定度，把难例交给专家。我们要回答的问题是：每一百张图只能复核十张时，这种方法能多找出多少原本会漏掉的青光眼？

## Slide 2: Fundus images and data splits

### English

We have about 7,700 fundus images, split between glaucoma and normal eyes. The table shows the exact counts for each source. We will set aside the 705 ACRIMA images for the final test. This leaves about 7,000 images for model development and internal testing. Before splitting them, we will group each original photo with its edited or augmented versions. ODIR adds a separate set of 3,500 patients with photos of both eyes. We will keep each patient's two eyes in the same split. These groups let us compare methods on the same test images and trace every image back to its source.

### 中文理解稿

现有数据约有 7,700 张眼底图，包括青光眼和正常眼底，表格列出了各来源的具体数量。我们会留出 ACRIMA 的 705 张图做最终测试，剩余约 7,000 张用于模型开发和内部测试。划分前，先把原图及其编辑、增强版本放在同一组。ODIR 还提供了 3,500 位患者的双眼照片，同一患者的两只眼也会放进同一个集合。各方法共用这些分组和测试图，每张图都保留来源记录。

## Slide 3: Screening and referral workflow

### English

Here is how the system works. First, it checks whether the photo is sharp, well exposed and clear enough to assess. If it needs a retake, the screen gives a reason, such as poor focus or low brightness. Next, a small model called MobileNetV3 estimates the probability of glaucoma. We calibrate these probabilities using a separate set of images. The system then combines image quality and model uncertainty to choose cases for expert review. Clear photos with a low glaucoma probability receive a negative screening result. Positive screens and photos that remain unreadable receive a referral record. We count the total number of patients needing follow-up.

### 中文理解稿

系统先检查照片是否清楚、曝光是否合适。需要重拍时，屏幕会说明原因，比如对焦不清或画面太暗。接着，MobileNetV3 估计青光眼概率，我们用单独的一组图像校准这些概率。系统再结合图像质量和模型的不确定度，选出需要专家复核的图片。质量合格、患病概率低且未入选复核的图片给出筛查阴性结果；筛查阳性和重拍后仍看不清的图片生成转诊记录。最后统计有多少患者需要后续处理。

## Slide 4: Comparing review methods

### English

We will compare three ways to choose images for review: random selection, model uncertainty, and our combined score. Each method uses the same model and can select ten out of every hundred images. We will also test five and twenty. The tests include original photos and versions with blur, brightness changes or lower contrast. The main question is how many glaucoma cases receive a negative screening result. We also ask how many negative results are correct. Our goal is more than 98 out of every hundred. Alongside this, we will report how many people receive negative results, how many errors expert review catches, and how many patients need a referral.

### 中文理解稿

我们比较三种选图方式：随机选择、按模型不确定度选择，以及两种信号结合。各方法使用同一个模型，每一百张图选十张复核，也会测试选五张和二十张的情况。测试包括原图，以及加入模糊、亮度变化和低对比度的版本。最关心的是有多少青光眼被判为筛查阴性，也会看阴性结果中有多少是正确的，目标是每一百个超过九十八个。同时报告阴性结果覆盖多少人、复核找出了多少错误，以及多少患者需要转诊。

## Slide 5: Offline demo and final submission

### English

The demo runs offline on a laptop. A health worker imports a batch of photos and sees three lists. Green shows negative screening results. Yellow asks for a retake and explains the image problem. Red shows positive screens, cases selected for expert review, and photos that remain unreadable. Each referral record includes the original photo, the glaucoma probability and the reason for referral. Our target is to process each image in under two hundred milliseconds, using less than five hundred megabytes of memory. The final submission will include the data lists, code, experiment results and demo. Our four team members will share the data, model, evaluation and demo work.

### 中文理解稿

演示可以在笔记本上离线运行。工作人员导入一批照片后，会看到三个列表：绿色显示筛查阴性，黄色提示重拍并说明原因，红色显示筛查阳性、需要专家复核的病例，以及重拍后仍看不清的照片。每份转诊记录附上原图、青光眼概率和转诊原因。我们希望单张图处理时间低于 200 毫秒，内存使用低于 500 MB。最终提交数据清单、代码、实验结果和演示程序，四位成员分工完成数据、模型、评估和演示。

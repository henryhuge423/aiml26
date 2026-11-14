import json, pathlib, statistics, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

root=pathlib.Path(__file__).resolve().parents[3]
out=pathlib.Path(__file__).resolve().parent
runs={'A':root/'runs/formal_v3/glaucoma_a','B':root/'runs/formal_v3/glaucoma_b_candidate','Distilled quality':root/'runs/formal_v3/quality_student','EyeQ quality':root/'runs/eyeq_quality_v1/quality_student'}
def read(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
snapshot=out/'chart_data.json'
if snapshot.exists():
 raw=json.loads(snapshot.read_text(encoding='utf8'))
 data={k:(v['steps'],v['epochs']) for k,v in raw.items()}
else:
 data={k:([{'step':x['step'],'epoch':x['epoch'],'loss':x['loss']} for x in read(p/'steps.jsonl')],read(p/'epochs.jsonl')) for k,p in runs.items()}
 snapshot.write_text(json.dumps({k:{'steps':s,'epochs':e} for k,(s,e) in data.items()},separators=(',',':')),encoding='utf8')
for name,(steps,epochs) in data.items():
 assert len(epochs)==20 and [x['step'] for x in steps]==list(range(1,len(steps)+1))
 assert all(math.isfinite(x['loss']) for x in steps)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':200})
colors={'A':'#176a8a','B':'#e07a36','Distilled quality':'#6b5b95','EyeQ quality':'#18876e'}
fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for k in ['A','B']:
 s,e=data[k]; means=[statistics.mean(x['loss'] for x in s if x['epoch']==i) for i in range(20)]
 axs[0].plot(range(1,21),means,label=k,color=colors[k],linewidth=2)
 axs[1].plot(range(1,21),[x['loss'] for x in e],label=k,color=colors[k],linewidth=2)
for ax,title in zip(axs,['Training loss (epoch mean)','Validation loss']):
 ax.set(title=title,xlabel='Epoch',ylabel='Binary cross-entropy');ax.grid(alpha=.2);ax.legend();ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.suptitle('Glaucoma models: loss across training',weight='bold');fig.savefig(out/'01_glaucoma_loss.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
sources=[('airogs-light-v2','AIROGS (n=769; selection source)'),('DATASET_101/ODIR-5K','ODIR-5K (n=277)'),('Kaggle-Eye_diseases_classification','Kaggle (n=144)')]
for ax,(src,title) in zip(axs,sources):
 for k in ['A','B']:
  e=data[k][1];ax.plot(range(1,21),[x['per_source'][src]['auc'] for x in e],label=k,color=colors[k],linewidth=2)
 ax.set(title=title,xlabel='Epoch',ylim=(.75,1.01));ax.grid(alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylabel('Validation ROC AUC');axs[0].legend();fig.suptitle('Glaucoma models: validation AUC by source',weight='bold');fig.savefig(out/'02_glaucoma_auc.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
e=data['Distilled quality'][1];axs[0].plot(range(1,21),[x['loss'] for x in e],color=colors['Distilled quality']);axs[0].set(title='Distilled quality',ylabel='Validation Huber loss')
e=data['EyeQ quality'][1];axs[1].plot(range(1,21),[x['teacher_mae'] for x in e],color=colors['EyeQ quality']);axs[1].set(title='EyeQ quality: teacher agreement',ylabel='Validation MAE to teacher')
acc=[sum(x['eyeq_confusion'][i][i] for i in range(3))/x['eyeq_n'] for x in e]
axs[2].plot(range(1,21),acc,label='Accuracy',color=colors['EyeQ quality']);axs[2].plot(range(1,21),[x['eyeq_macro_f1'] for x in e],label='Macro F1',color='#dc8346');axs[2].set(title='EyeQ human labels (n=117)',ylabel='Validation score',ylim=(0,1.02));axs[2].legend()
for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.suptitle('Quality models: validation metrics',weight='bold');fig.savefig(out/'03_quality_validation.png');plt.close(fig)
for k,(s,e) in data.items():
 print(k, 'steps',len(s),'epoch1 mean',statistics.mean(x['loss'] for x in s if x['epoch']==0),'epoch20 mean',statistics.mean(x['loss'] for x in s if x['epoch']==19))
 if k in ['A','B']:
  best=max(e,key=lambda x:x['selection_value']);print('best selection epoch',best['epoch']+1,'auc',best['selection_value'],'loss',best['loss'])
 elif k=='EyeQ quality':
  best=min(e,key=lambda x:x['selection_loss']);print('best epoch',best['epoch'],'selection',best['selection_loss'],'accuracy',sum(best['eyeq_confusion'][i][i] for i in range(3))/best['eyeq_n'],'f1',best['eyeq_macro_f1'],'teacher mae',best['teacher_mae']);print('first last acc',acc[0],acc[-1])
 else:
  best=min(e,key=lambda x:x['loss']);print('best epoch',best['epoch']+1,'loss',best['loss'])

# Moving averages show the step-level record without hiding its scale.
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,(k,(steps,epochs)) in zip(axs.flat,data.items()):
 window=100
 xs=[steps[i]['step'] for i in range(window-1,len(steps),window)]
 ys=[statistics.mean(x['loss'] for x in steps[i-window+1:i+1]) for i in range(window-1,len(steps),window)]
 ax.plot(xs,ys,color=colors[k],linewidth=1.2);ax.set(title=k,xlabel='Training step',ylabel='Mean loss per 100 steps');ax.grid(alpha=.2)
fig.suptitle('Training loss throughout each run',weight='bold');fig.savefig(out/'04_step_loss.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for k in ['Distilled quality','EyeQ quality']:
 s,e=data[k];axs[0].plot(range(1,21),[statistics.mean(x['loss'] for x in s if x['epoch']==i) for i in range(20)],label=k,color=colors[k])
axs[0].set(title='Training loss',xlabel='Epoch',ylabel='Mean batch loss');axs[0].legend()
e=data['Distilled quality'][1]
for src,label in [('airogs-light-v2','AIROGS'),('DATASET_101/ODIR-5K','ODIR-5K'),('Kaggle-Eye_diseases_classification','Kaggle'),('fiqs','FIQS')]:
 axs[1].plot(range(1,21),[x['per_source'][src]['mae_to_teacher'] for x in e],label=label)
axs[1].set(title='Distilled quality: error by source',xlabel='Epoch',ylabel='Validation MAE to teacher');axs[1].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.suptitle('Quality models: training and validation',weight='bold');fig.savefig(out/'05_quality_detail.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
e=data['EyeQ quality'][1]
for ax,rec,title in [(axs[0],e[16],'Selected checkpoint: epoch 17'),(axs[1],e[19],'Last checkpoint: epoch 20')]:
 matrix=rec['eyeq_confusion'];ax.imshow(matrix,cmap='Blues',vmin=0,vmax=73)
 for i in range(3):
  for j in range(3):ax.text(j,i,str(matrix[i][j]),ha='center',va='center',color='white' if matrix[i][j]>36 else '#17324d',fontsize=12)
 ax.set(title=title,xlabel='Predicted grade',ylabel='Human grade',xticks=range(3),yticks=range(3),xticklabels=['0','1','2'],yticklabels=['0','1','2'])
fig.suptitle('EyeQ validation: three-grade confusion matrices (n=117)',weight='bold');fig.savefig(out/'06_eyeq_confusion.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
labels=['Glaucoma A / B','Distilled quality','EyeQ quality'];train=[10354,12147,4722];val=[1190,1642,616]
y=range(3);ax.barh(y,train,label='Training',color='#176a8a');ax.barh(y,val,left=train,label='Validation',color='#e6a04b')
for i,(a,b) in enumerate(zip(train,val)):ax.text(a/2,i,f'{a:,}',ha='center',va='center',color='white');ax.text(a+b+130,i,f'{b:,} validation',va='center')
ax.set(yticks=list(y),yticklabels=labels,xlabel='Number of images',xlim=(0,15500));ax.invert_yaxis();ax.legend(loc='lower right');ax.set_title('Training and validation set sizes',weight='bold');fig.savefig(out/'07_dataset_sizes.png');plt.close(fig)

"""Regenerate figures from downloaded Drive logs. Requires numpy and matplotlib."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
logs = json.loads((ROOT / 'figure_source_logs.json').read_text(encoding='utf-8'))
for r in logs:
    h=r['data']['history']
    if 'epochs' not in h: h['epochs']=h['epoch']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def onset(h, key, threshold):
    return next((ep for ep, acc in zip(h['epochs'], h[key]) if acc >= threshold), None)

rows=[]
for r in logs:
    h,s=r['data']['history'],r['data']['summary']
    e=np.array(h['epochs']); tr=np.array(h['train_acc']); te=np.array(h['test_acc'])
    assert len(e)==len(tr)==len(te) and np.all(np.diff(e)>0)
    assert np.all((tr>=0)&(tr<=1)) and np.all((te>=0)&(te<=1))
    t99=onset(h,'test_acc',.99); train99=onset(h,'train_acc',.99)
    assert t99==s['test_epoch_99'] and train99==s['train_epoch_99']
    rows.append(dict(folder=r['folder'],seed=s['seed'],weight_decay=s['weight_decay'],lr=s['lr'],train_99=train99,test_99=t99,gap_99=t99-train99,logging_interval=int(np.min(np.diff(e))),last_epoch=int(e[-1]),source_url=r['url']))
with (ROOT/'verified_onsets.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

groups=[('diag_layernorm_raw_lr_6e-4_long','LayerNorm / raw head',1),('diag_sphere_bounded_lr_6e-4','Sphere / bounded head',1),('diag_sphere_bounded_lr_6e-4_wd0_long','Sphere / bounded head',0)]
colors=['#0072B2','#D55E00','#009E73']
fig,axes=plt.subplots(2,3,figsize=(10.5,5.5),sharey=True)
stats=[]
for col,(folder,title,wd) in enumerate(groups):
    rr=sorted([r for r in logs if r['folder']==folder],key=lambda r:r['data']['summary']['seed'])
    assert [r['data']['summary']['seed'] for r in rr]==[1,2,3]
    vals=[onset(r['data']['history'],'test_acc',.99) for r in rr]
    stats.append({'condition':title,'wd':wd,'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))})
    for row,xmax in enumerate([9000,2000]):
        ax=axes[row,col]
        for r,color in zip(rr,colors):
            s,h=r['data']['summary'],r['data']['history']
            assert s['weight_decay']==wd and s['lr']==.0006
            ax.plot(h['epochs'],np.array(h['train_acc'])*100,color=color,ls='--',lw=1,alpha=.65)
            ax.plot(h['epochs'],np.array(h['test_acc'])*100,color=color,lw=1.6)
        ax.set(xlim=(0,xmax),ylim=(-2,104),xlabel='Epoch',yticks=[0,25,50,75,100])
        ax.grid(alpha=.15)
        ax.set_title(f"({'abcdef'[row*3+col]}) {title}\nWD = {wd}",fontsize=10)
        if col==0: ax.set_ylabel(('Full view' if row==0 else 'Early view')+'\nAccuracy (%)')
        if row==0: ax.text(.97,.08,f'Test-99: {np.mean(vals):,.0f} ± {np.std(vals,ddof=1):,.0f}',transform=ax.transAxes,ha='right',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
handles=[Line2D([0],[0],color=c,label=f'Seed {i}') for i,c in enumerate(colors,1)]+[Line2D([0],[0],color='#333333',ls='--',label='Train'),Line2D([0],[0],color='#333333',label='Test')]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.005))
fig.tight_layout(rect=(0,.065,1,1))
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'figure1_corrected.{ext}',dpi=300)
plt.close(fig)

fig,ax=plt.subplots(figsize=(5.8,3.7))
for cond,label,color,offset in [('layernorm_raw','LayerNorm / raw head','#0072B2',-.045),('sphere_bounded','Sphere / bounded head','#D55E00',.045)]:
    means=[];sds=[]
    for i,lr in enumerate([.0001,.0003,.0006,.001]):
        rr=[r for r in logs if '__lr' in r['folder'] and r['data']['summary']['condition']==cond and r['data']['summary']['lr']==lr]
        assert len(rr)==3 and all(r['data']['summary']['weight_decay']==1 for r in rr)
        vals=[onset(r['data']['history'],'test_acc',.99) for r in rr]
        means.append(np.mean(vals));sds.append(np.std(vals,ddof=1))
        ax.scatter(np.full(3,i+offset),vals,color=color,alpha=.5,s=18,zorder=3)
    ax.errorbar(np.arange(4)+offset,means,yerr=sds,color=color,marker='o',capsize=4,lw=1.4,label=label)
ax.set(yscale='log',xlabel='Learning rate',ylabel='Epoch at first test accuracy ≥99%',xticks=range(4),xticklabels=['1e−4','3e−4','6e−4','1e−3'])
ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'learning_rate_control.{ext}',dpi=300)
plt.close(fig)
print(json.dumps(stats,indent=2))
print('Validated 33 runs; generated two figures in PNG, PDF, SVG.')

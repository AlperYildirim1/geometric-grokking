from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).resolve().parent/'camera_ready_review'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
x=np.arange(4)
fig,axes=plt.subplots(1,2,figsize=(7,2.65))
data=[([57317,15033,6900,3917],[8115,3303,1820,1568]),([44567,12353,5767,3033],[717,-338,-137,-57])]
for ax,(learned,uniform) in zip(axes,data):
    ax.plot(x,learned,'o-',color='#237bb5',lw=1.2,ms=4,label='Learned attention')
    ax.plot(x,uniform,'s-',color='#d55e00',lw=1.2,ms=4,label='Uniform attention')
    ax.set_xticks(x,['$10^{-4}$','$3\\times10^{-4}$','$6\\times10^{-4}$','$10^{-3}$'])
    ax.set_xlabel('Learning rate'); ax.grid(alpha=.18)
axes[0].set_yscale('log'); axes[0].set_ylabel('Test-99 epoch'); axes[0].set_title('(a) Completion')
axes[0].set_yticks([1000,3000,10000,30000,60000],['1,000','3,000','10,000','30,000','60,000'])
axes[1].set_yscale('symlog',linthresh=1000); axes[1].set_ylabel('Rise onset minus Train-99'); axes[1].set_title('(b) Post-memorization waiting')
axes[1].set_yticks([-500,0,1000,10000,50000],['−500','0','1,000','10,000','50,000'])
axes[1].axhline(0,color='.4',lw=.7,ls='--')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.015))
fig.tight_layout(rect=(0,.11,1,1))
for ext in ['pdf','png','svg']:fig.savefig(out/f'uniform_attention_summary.{ext}',dpi=220,bbox_inches='tight')
print('Uniform-attention summary figure saved from rebuttal aggregate means; no synthetic per-seed data.')

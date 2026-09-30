import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
AI,HU,EX='#2a78d6','#eb6834','#1baf7a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e6','savefig.dpi':220,'savefig.bbox':'tight'})
z=np.load('mc.npz');rho,v,win,b,rounds=z['rho'],z['v'],z['win'],z['b'],z['rounds'];cur=np.load('cur.npy')
f,ax=plt.subplots(1,3,figsize=(12,3.4),gridspec_kw={'width_ratios':[1.1,1,1.1],'wspace':0.55})
vb=np.linspace(.7,.98,15);rb=np.linspace(.1,.6,11);H=np.full((len(rb)-1,len(vb)-1),np.nan)
for i in range(len(rb)-1):
  for j in range(len(vb)-1):
    s=(rho>=rb[i])&(rho<rb[i+1])&(v>=vb[j])&(v<vb[j+1]); H[i,j]=win[s].mean()
im=ax[0].imshow(H,origin='lower',aspect='auto',extent=[.7,.98,.1,.6],cmap='Blues',vmin=0,vmax=1)
ax[0].set_xlabel('Verifier accuracy v');ax[0].set_ylabel('Error correlation ρ');ax[0].set_title('(a) P(swarm wins), equal cost',loc='left',fontweight='bold');ax[0].grid(False)
plt.colorbar(im,ax=ax[0],fraction=.046);ax[0].axvline(.96,color=HU,ls='--',lw=1);ax[0].text(.955,.57,'frontier per-step\naccuracy ≈ 0.96',ha='right',fontsize=7,color=HU)
names=['small-model p','frontier p','error corr ρ','verifier v','steps m','cost ratio'];o=np.argsort(b)
ax[1].barh(range(6),b[o],color=[AI if x>0 else EX for x in b[o]]);ax[1].set_yticks(range(6));ax[1].set_yticklabels([names[i] for i in o]);ax[1].axvline(0,color='k',lw=.7)
ax[1].set_title('(b) Drivers of the gap',loc='left',fontweight='bold');ax[1].set_xlabel('Standardised regression coeff.');ax[1].grid(axis='y',visible=False)
win0=win.mean();gen=[((win)|(rounds<=k)).mean() for k in range(1,21)]
ax[2].plot(range(0,21),[win0]+gen,color=HU,lw=2,label='Self-improve generator only');ax[2].plot(range(0,21),[win0]+list(cur),color=AI,lw=2,label='Self-improve generator + verifier')
ax[2].set_xlabel('Self-improvement rounds');ax[2].set_ylabel('Share of scenarios swarm wins');ax[2].set_ylim(0,1);ax[2].legend(fontsize=7,loc='upper left');ax[2].set_title('(c) Self-improvement target',loc='left',fontweight='bold')
f.savefig('agentic_mc.png');print('ok')

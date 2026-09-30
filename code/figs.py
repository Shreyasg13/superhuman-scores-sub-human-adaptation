import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
import datetime as dt
AI,HU,EX,AID,AIS,INK,INK2,GRID='#2a78d6','#eb6834','#1baf7a','#1c5cab','#86b6ef','#1a1a1a','#555555','#e6e6e6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#999','axes.linewidth':.6,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False,'xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK,'axes.titlesize':10,'axes.titleweight':'bold','savefig.dpi':300,'savefig.bbox':'tight','legend.frameon':False})
def save(f,n): f.savefig(f'fig/{n}.pdf'); f.savefig(f'fig/{n}.png',dpi=200); plt.close(f)

# F1 bell
f,ax=plt.subplots(figsize=(6.4,2.8));x=np.linspace(50,160,600);y=norm.pdf(x,100,15)*100
ax.fill_between(x,y,color=HU,alpha=.22,lw=0);ax.plot(x,y,color=HU,lw=1.6,label='Human population (mean 100, SD 15)')
for v,c,l,yy in [(100,'#888','Human mean',2.75),(130,'#888','Mensa ≈130',2.45),(135,AID,'Top AI, offline test: 135\n(≈ top 1%)',2.05),(151,AI,'Top AI, public test: 151\n(≈ 1 in 2,970)',1.35)]:
    ax.axvline(v,color=c,lw=1.8 if c!='#888' else .9,ls='-' if c!='#888' else '--');ax.text(v+1 if v<151 else v-1,yy,l,ha='left' if v<151 else 'right',va='top',fontsize=7.5,color=INK)
ax.set_xlim(50,160);ax.set_ylim(0,2.9);ax.set_xlabel('IQ score');ax.set_ylabel('% of population per IQ point');ax.legend(loc='center left',fontsize=7.5,bbox_to_anchor=(0,.55))
save(f,'f1_bell')

# F2 IQ bars
M=[('Claude Fable 5.1',151,130),('GPT-6 Astra Ultra (Vision)',151,None),('Claude Fable 5.1 (Vision)',145,None),('GPT-5.6 Sol Ultra',145,130),('Gemini 3.1 Pro',145,None),('Grok-4.20 Expert (Apr)',145,None),('Claude Opus 5',144,129),('Claude Opus 5 MAX',142,127),('GPT-5.6 Sol Ultra (Vision)',133,131),('GPT-5.6 Terra Ultra (Vision)',None,135)]
f,ax=plt.subplots(figsize=(6.4,3.6));yy=np.arange(len(M))[::-1];h=.38
ax.barh(yy+h/2,[m[1] or 0 for m in M],h,color=AIS,label='Mensa Norway (public)')
ax.barh(yy-h/2,[m[2] or 0 for m in M],h,color=AID,label='Offline test (never online)')
for i,m in enumerate(M):
    for off,v in ((h/2,m[1]),(-h/2,m[2])):
        if v: ax.text(v+.6,yy[i]+off,str(v),va='center',fontsize=7,color=INK)
ax.axvline(100,color=HU,ls='--',lw=1);ax.axvline(130,color=HU,ls=':',lw=1)
ax.text(100.5,len(M)-.35,'human mean',fontsize=7,color=HU);ax.text(130.5,len(M)-.35,'Mensa',fontsize=7,color=HU)
ax.set_yticks(yy);ax.set_yticklabels([m[0] for m in M],fontsize=7.8);ax.set_xlim(90,158);ax.set_xlabel('IQ score');ax.grid(axis='y',visible=False)
ax.legend(loc='lower center',fontsize=7.5,bbox_to_anchor=(.4,1.0),ncol=2);save(f,'f2_iq')

# F3 ARC
f,(a,b)=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw={'width_ratios':[1.25,1]})
lab=['ARC-AGI-1','ARC-AGI-2','ARC-AGI-3'];xx=np.arange(3);w=.26
a.bar(xx-w,[98.5,95,62.7],w,color=AI,label='Best AI');a.bar(xx,[0,66,0],w,color=HU,label='Average human');a.bar(xx+w,[0,100,100],w,color=EX,label='Human panel / baseline')
for i,v in enumerate([98.5,95,62.7]): a.text(i-w,v+1.5,f'{v:g}',ha='center',fontsize=6.8)
a.text(2,4,'n/a',ha='center',fontsize=6.5,color=INK2);a.text(0+w/2,4,'no verified\nhuman data',ha='center',fontsize=5.8,color=INK2)
a.set_xticks(xx);a.set_xticklabels(lab);a.set_ylim(0,112);a.set_ylabel('Score (%)');a.set_title('(a) Best score, Sep 2026',loc='left');a.legend(fontsize=6.8,loc='upper center',ncol=3,bbox_to_anchor=(.9,-.12));a.grid(axis='x',visible=False)
dts=['Mar 26','Jul 26','Sep 26'];vals=[0.37,30.2,62.7]
b.plot(range(3),vals,color=AI,marker='o',lw=1.8,ms=5);b.fill_between(range(3),vals,color=AI,alpha=.12)
for i,(v,n) in enumerate(zip(vals,['Gemini 3.1 Pro','Claude Opus 5','GPT-6 Astra'])): b.text(i,v+5,f'{v:g}%\n{n}',ha='center',fontsize=6.6)
b.axhline(100,color=EX,ls='--',lw=1);b.text(0,102,'human baseline 100%',fontsize=6.8,color=EX)
b.set_xticks(range(3));b.set_xticklabels(dts);b.set_ylim(0,115);b.set_xlim(-.4,2.4);b.set_title('(b) ARC-AGI-3 frontier, 2026',loc='left')
save(f,'f3_arc')

# F4 radar
D=[("Knowledge\nbreadth",97,45,70),("Static abstract\nreasoning",92,50,95),("Expert problem\nsolving",88,20,95),("Novel interactive\nlearning",60,85,100),("Long-horizon\nautonomy",45,65,95),("Sample\nefficiency",30,75,90),("Continual learning\n& memory",15,80,90),("Speed &\nscale",99,5,10),("Embodied &\nsocial judgment",20,85,90)]
ang=np.linspace(0,2*np.pi,len(D),endpoint=False).tolist();ang+=ang[:1]
f=plt.figure(figsize=(5.2,4.6));ax=f.add_subplot(111,polar=True)
for j,c,l,ls in [(1,AI,'Frontier AI (best of top 10)','-'),(2,HU,'Average adult','-'),(3,EX,'Top human expert (own field)','--')]:
    v=[d[j] for d in D]+[D[0][j]];ax.plot(ang,v,color=c,lw=1.8,ls=ls,label=l);ax.fill(ang,v,color=c,alpha=.10 if j<3 else 0)
ax.set_xticks(ang[:-1]);ax.set_xticklabels([d[0] for d in D],fontsize=7.2);ax.set_ylim(0,100);ax.set_yticks([25,50,75,100]);ax.set_yticklabels(['25','50','75','100'],fontsize=6.5,color=INK2)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.06),ncol=3,fontsize=7);save(f,'f4_radar')

# F5 weighting
P={'IQ-test':[3,5,4,1,0,0,0,1,0],'Adaptability\n(Chollet)':[0,2,1,5,2,5,4,0,1],'Economic\nwork':[3,2,3,2,5,2,3,4,3],'Balanced\n(CHC-style)':[1]*9}
f,axs=plt.subplots(1,2,figsize=(6.6,2.8),sharey=True);xx=np.arange(4);w=.26
for ai,(ax,kind) in enumerate(zip(axs,['Arithmetic $I_A$','Bottleneck (geometric) $I_G$'])):
    for j,(c,l) in enumerate([(AI,'Frontier AI'),(HU,'Average adult'),(EX,'Top human expert')]):
        vals=[]
        for wv in P.values():
            wv=np.array(wv,float);s=np.array([d[j+1] for d in D],float)
            vals.append((wv*s).sum()/wv.sum() if ai==0 else np.exp((wv/wv.sum()*np.log(np.maximum(s,1))).sum()))
        ax.bar(xx+(j-1)*w,vals,w,color=c,label=l)
        for i,v in enumerate(vals): ax.text(i+(j-1)*w,v+1.2,f'{v:.0f}',ha='center',fontsize=6)
    ax.set_xticks(xx);ax.set_xticklabels(list(P.keys()),fontsize=7);ax.set_title(f'({"ab"[ai]}) {kind}',loc='left');ax.set_ylim(0,105);ax.grid(axis='x',visible=False)
axs[0].set_ylabel('Composite score (0–100)');axs[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(1.0,-.2),ncol=3);save(f,'f5_weights')

# F6 3D landscape
sig=lambda x:1/(1+np.exp(-x))
Pf=lambda n,h,H,k,b:sig(k*np.log2(H/h)-b*(n-.2))
n=np.linspace(0,1,41);Y=np.linspace(np.log2(1/60),np.log2(2000),61);N,YY=np.meshgrid(n,Y);h=2**YY
f=plt.figure(figsize=(6.4,4.6));ax=f.add_subplot(111,projection='3d')
ZA=Pf(N,h,31.8,.574,6.0);ZM=Pf(N,h,2000,.5,1.5)
ax.plot_surface(N,YY,ZA,cmap='Blues',vmin=-.3,vmax=1.1,alpha=.95,linewidth=0,antialiased=True,rstride=1,cstride=1)
ax.plot_surface(N,YY,ZM,cmap='Greens',vmin=-.3,vmax=1.2,alpha=.45,linewidth=0,antialiased=True)
for H_,k,b,c in [(31.8,.574,6.0,AID),(2000,.5,1.5,'#0b6b4a')]:
    yl=np.log2(H_)-b*(n-.2)/k;m=(yl>=Y[0])&(yl<=Y[-1]);ax.plot(n[m],yl[m],.5,color=c,lw=2.2)
tv=[1/60,1,8,40,167,2000];ax.set_yticks(np.log2(tv));ax.set_yticklabels(['1 min','1 h','1 day','1 wk','1 mo','1 yr'],fontsize=7)
ax.set_xlabel('Task novelty $n$',fontsize=8);ax.set_ylabel('Expert task length $h$',fontsize=8,labelpad=6);ax.set_zlabel('P(success)',fontsize=8)
ax.set_zlim(0,1);ax.view_init(elev=24,azim=-58);ax.tick_params(labelsize=7)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#3a7fc8',label='Frontier AI (Sep 2026, $H_{50}$ = 32 h)'),Patch(color='#7cc49a',label='Human master ($H_{50}$ = 2,000 h)')],loc='upper left',fontsize=7.5)
save(f,'f6_landscape')

# F7 hexbin
rng=np.random.default_rng(20260927);g1=rng.standard_normal(20000);g2=rng.standard_normal(20000);px=g1;py=.6*g1+.8*g2
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('o',['#fde8dd','#f4a07a','#e0662f','#8f310c'])
f=plt.figure(figsize=(6.2,5.4));gs=f.add_gridspec(2,2,width_ratios=[5,1],height_ratios=[1,5],hspace=.04,wspace=.04)
ax=f.add_subplot(gs[1,0]);ht=f.add_subplot(gs[0,0],sharex=ax);hr=f.add_subplot(gs[1,1],sharey=ax)
ax.hexbin(px,py,gridsize=34,extent=(-4,4,-4,4),cmap=cm,mincnt=1,linewidths=.2,edgecolors='white')
ht.hist(px,bins=40,range=(-4,4),color=HU,edgecolor='white',lw=.4);hr.hist(py,bins=40,range=(-4,4),color=HU,edgecolor='white',lw=.4,orientation='horizontal')
for a_ in (ht,hr): a_.axis('off')
zs=lambda iq:(iq-100)/15; za=lambda s:norm.ppf(s/200)
pts=[('Claude Opus 5',zs(129),za(30.2),1),('GPT-5.6 Sol',zs(130),za(7.8),1),('GPT-6 Astra',zs(130),za(62.7),1),('Grok 4.6 (est.)',zs(127),za(2.1),0),('Gemini 3.1 Pro, Mar (est.)',zs(127),za(0.37),0)]
for nm,x_,y_,m in pts:
    ax.scatter(x_,y_,s=46,color=AI if m else 'white',edgecolor=AI if not m else 'white',linewidth=1.6 if not m else .8,zorder=5)
    ax.text(x_+.13,y_,nm,fontsize=7,va='center',zorder=6)
ax.scatter(3,3,marker='*',s=220,color=EX,edgecolor='white',zorder=6);ax.text(2.85,2.62,'Human master (z = 3)',fontsize=7,ha='right')
ax.scatter(0,0,marker='D',s=40,color=HU,edgecolor='white',zorder=6);ax.text(.15,-.28,'Average adult',fontsize=7)
ax.axhline(0,color='#999',ls='--',lw=.7);ax.axvline(0,color='#999',ls='--',lw=.7)
for (x_,y_,t,ha) in [(-3.9,3.7,'FAST LEARNER','left'),(3.9,3.7,'MASTER ZONE','right'),(-3.9,-3.8,'DEVELOPING','left'),(3.9,-3.8,'TEST-TAKER CORNER','right')]: ax.text(x_,y_,t,fontsize=6.5,color='#888',ha=ha,family='DejaVu Sans Mono')
ax.set_xlim(-4,4);ax.set_ylim(-4,4);ax.set_xlabel('Static reasoning  $z_s=(IQ_{off}-100)/15$');ax.set_ylabel('Adaptive learning  $z_a=\\Phi^{-1}(S_{ARC3}/2)$')
save(f,'f7_density')

# F8 METR
T0=dt.date(2026,5,8)
yr=lambda d:d.year+(d.timetuple().tm_yday-1)/365.25
hist=[('GPT-2',dt.date(2019,2,1),2/60),('GPT-3',dt.date(2020,5,1),9/60),('GPT-3.5',dt.date(2022,3,1),.6),('GPT-4',dt.date(2023,3,1),4),('GPT-4o',dt.date(2024,5,1),6),('Claude 3.5 Sonnet (new)',dt.date(2024,10,22),20),('o1',dt.date(2024,12,5),38),('Claude 3.7 Sonnet',dt.date(2025,2,24),60),('Claude Opus 4.5',dt.date(2025,11,24),300),('Claude Opus 4.6',dt.date(2026,2,5),719),('Claude Mythos Preview (≥)',dt.date(2026,5,8),960)]
f,ax=plt.subplots(figsize=(6.6,3.4))
ax.plot([yr(d) for _,d,_ in hist],[m/60 for *_,m in hist],'o-',color=AI,ms=4.5,lw=1.5,label='Measured 50% horizon (METR)')
ts=np.array([T0+dt.timedelta(days=int(d)) for d in np.arange(0,970,7)]);tx=[yr(t) for t in ts];dd=np.arange(0,970,7)
ax.fill_between(tx,16*2**(dd/200),16*2**(dd/100),color=AI,alpha=.15,lw=0,label='Doubling 100–200 days')
ax.plot(tx,16*2**(dd/131),'--',color=AI,lw=1.4,label='Projection, 131-day doubling')
for v,l in [(8,'1 workday'),(40,'1 work-week'),(167,'1 work-month'),(2000,"master's 1-year project")]:
    ax.axhline(v,color=HU,ls=':',lw=.9);ax.text(2019.05,v*1.12,l,fontsize=6.8,color=HU)
for nm,d,m in hist:
    if nm in ('GPT-2','GPT-4','o1','Claude Opus 4.6','Claude Mythos Preview (≥)'): ax.annotate(nm,(yr(d),m/60),textcoords='offset points',xytext=(4,-10 if nm!='Claude Mythos Preview (≥)' else 6),fontsize=6.5,ha='left' if nm!='Claude Mythos Preview (≥)' else 'right')
ax.axvline(yr(dt.date(2026,9,27)),color='#999',lw=.8);ax.text(yr(dt.date(2026,9,27))+.03,3e-4,'27 Sep 2026',fontsize=6.5,color='#777')
ax.set_yscale('log');ax.set_ylim(2e-4,2e4);ax.set_xlim(2019,2029)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:{1e-3:'3.6 s',1e-2:'36 s',1e-1:'6 min',1:'1 h',10:'10 h',100:'100 h',1000:'1,000 h',1e4:'10,000 h'}.get(v,'')))
ax.set_ylabel('50% time horizon (expert time)');ax.legend(fontsize=7,loc='lower right');save(f,'f8_metr')

# F9 IQ* vs lambda
f,ax=plt.subplots(figsize=(5.6,2.9));L=np.linspace(0,1,50)
for nm,a_,b_,c,ls in [('Human master',3,3,EX,'-'),('Average adult',0,0,HU,'-'),('GPT-6 Astra',zs(130),za(62.7),AI,'-'),('Claude Opus 5',zs(129),za(30.2),AI,'--'),('GPT-5.6 Sol',zs(130),za(7.8),AI,':')]:
    v=100+15*((1-L)*a_+L*b_);ax.plot(L,v,color=c,ls=ls,lw=1.8,label=nm)
ax.axvline(.5,color='#999',lw=.7);ax.set_xlabel('Weight on adaptive learning  $\\lambda$');ax.set_ylabel('$IQ^*$');ax.set_xlim(0,1)
ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1,.5));save(f,'f9_iqstar')
print('done')

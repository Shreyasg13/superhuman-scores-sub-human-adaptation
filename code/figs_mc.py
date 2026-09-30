import numpy as np, matplotlib, datetime as dt
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('figs.py').read().split("# F1 bell")[0])
D=np.load('mc_D.npy');Da=np.load('mc_Davg.npy');src=np.load('mc_src.npy')
names=["Knowledge","Static reasoning","Expert problems","Novel learning","Long-horizon","Sample efficiency","Continual learning","Speed & scale","Embodied/social"]
f,(a,b)=plt.subplots(1,2,figsize=(6.8,2.9),gridspec_kw={'width_ratios':[1.15,1]})
bins=np.linspace(-60,60,97)
a.hist(Da,bins=bins,color=HU,alpha=.75,label=f'AI − average adult  (AI ahead {np.mean(Da>0):.0%})')
a.hist(D,bins=bins,color=EX,alpha=.75,label=f'AI − top expert  (AI ahead {np.mean(D>0):.0%})')
a.axvline(0,color=INK,lw=.9);a.set_xlabel('Composite difference (points)');a.set_ylabel('Weightings (of 200,000)')
a.set_title('(a) Random weightings, Dirichlet(1)',loc='left');a.legend(fontsize=6.6,loc='upper left');a.grid(axis='x',visible=False)
o=np.argsort(src);cols=[AI if s>0 else EX for s in src[o]]
b.barh(range(9),src[o],color=cols);b.set_yticks(range(9));b.set_yticklabels([names[i] for i in o],fontsize=7)
b.axvline(0,color=INK,lw=.8);b.set_xlabel('Standardized regression coefficient');b.set_title('(b) Which weight moves the gap',loc='left');b.grid(axis='y',visible=False)
b.text(.02,.02,'← favours expert   favours AI →',transform=b.transAxes,fontsize=6.5,color=INK2)
save(f,'f10_mc_weights')
T0=dt.date(2026,5,8)
f,ax=plt.subplots(figsize=(6.4,2.6))
for h,c,l in [(40,AIS,'1 work-week (40 h)'),(167,AI,'1 work-month (167 h)'),(2000,AID,"master's 1-year project (2,000 h)")]:
    d=np.load(f'mc_days_{h}.npy');yrs=2026+(T0.timetuple().tm_yday+d)/365.25
    ax.hist(yrs,bins=np.linspace(2026.4,2030,110),color=c,alpha=.8,label=l)
    p=np.percentile(yrs,[10,50,90]);ax.plot([p[0],p[2]],[-300,-300],color=c,lw=3,solid_capstyle='butt');ax.plot(p[1],-300,'o',color=c,ms=4)
ax.set_xlim(2026.4,2030);ax.set_xlabel('Date reached');ax.set_ylabel('Simulations');ax.legend(fontsize=7);ax.grid(axis='x',visible=False)
save(f,'f11_mc_growth')
print('ok')

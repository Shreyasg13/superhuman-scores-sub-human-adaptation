import numpy as np, datetime as dt, json
S=np.array([[97,45,70],[92,50,95],[88,20,95],[60,85,100],[45,65,95],[30,75,90],[15,80,90],[99,5,10],[20,85,90]],float)
names=["Knowledge","Static reasoning","Expert problems","Novel learning","Long-horizon","Sample eff.","Continual learning","Speed & scale","Embodied/social"]
rng=np.random.default_rng(20260927); N=200000
out={}
def comp(W,Sx,geo):
    return np.exp(np.einsum('nd,ndk->nk',W,np.log(np.clip(Sx,1,100)))) if geo else np.einsum('nd,ndk->nk',W,Sx)
for alpha in (0.5,1,5):
    W=rng.dirichlet(np.full(9,alpha),N)
    for noise in (0,10):
        Sx=np.clip(S[None]+rng.normal(0,noise,(N,9,3)),1,100) if noise else np.broadcast_to(S,(N,9,3))
        for geo in (0,1):
            C=comp(W,Sx,geo)
            key=f"a{alpha}_n{noise}_{'geo' if geo else 'ari'}"
            out[key]=dict(ai_gt_avg=float((C[:,0]>C[:,1]).mean()),ai_gt_exp=float((C[:,0]>C[:,2]).mean()),
               ai_top=float(((C[:,0]>C[:,1])&(C[:,0]>C[:,2])).mean()),exp_top=float(((C[:,2]>C[:,0])&(C[:,2]>C[:,1])).mean()),
               d_exp_med=float(np.median(C[:,0]-C[:,2])),d_exp_p10=float(np.percentile(C[:,0]-C[:,2],10)),d_exp_p90=float(np.percentile(C[:,0]-C[:,2],90)),
               d_avg_med=float(np.median(C[:,0]-C[:,1])))
for k,v in out.items(): print(k,{a:round(b,3) for a,b in v.items()})
# sensitivity: standardized regression coefficients of (AI-expert) on weights (alpha=1, noise 0, arithmetic)
W=rng.dirichlet(np.ones(9),N);D=W@(S[:,0]-S[:,2])
Z=(W-W.mean(0))/W.std(0);src=np.linalg.lstsq(np.c_[np.ones(N),Z],(D-D.mean())/D.std(),rcond=None)[0][1:]
print('SRC',dict(zip(names,np.round(src,3))))
np.save('mc_D.npy',D);np.save('mc_Davg.npy',W@(S[:,0]-S[:,1]));np.save('mc_src.npy',src)
# growth MC
T0=dt.date(2026,5,8)
Td=np.exp(rng.normal(np.log(131),0.18,N)).clip(70,300); H0=rng.uniform(16,24,N)
res={}
for h in (40,167,2000):
    days=np.log2(h/H0)*Td; ds=[T0+dt.timedelta(days=float(x)) for x in np.percentile(days,[10,50,90])]
    res[h]=[str(d) for d in ds]; np.save(f'mc_days_{h}.npy',days)
print('growth',res, 'Td p10/50/90',np.percentile(Td,[10,50,90]).round())
# master lead share in Sep 2027/2028 under growth MC
sig=lambda x:1/(1+np.exp(-x))
n=np.linspace(0,1,21);y=np.linspace(np.log2(1/60),np.log2(2000),25);Nn,Y=np.meshgrid(n,y);h=2**Y
PM=sig(.5*np.log2(2000/h)-1.5*(Nn-.2))
for d in (dt.date(2027,9,15),dt.date(2028,9,15)):
    t=(d-T0).days;sh=[]
    for i in range(4000):
        H=H0[i]*2**(t/Td[i]);sh.append(((PM-sig(.574*np.log2(H/h)-6*(Nn-.2)))>0.25).mean())
    print(d,np.percentile(sh,[10,50,90]).round(3))
json.dump(out,open('mc.json','w'))

import numpy as np
from scipy.stats import norm
DIMS=[("Knowledge breadth",97,45,70),("Static abstract reasoning",92,50,95),("Expert problem solving",88,20,95),("Novel interactive learning",60,85,100),("Long-horizon autonomy",45,65,95),("Sample efficiency",30,75,90),("Continual learning & memory",15,80,90),("Speed & scale",99,5,10),("Embodied & social judgment",20,85,90)]
P={'IQ-test':[3,5,4,1,0,0,0,1,0],'Adaptability (Chollet)':[0,2,1,5,2,5,4,0,1],'Economic work':[3,2,3,2,5,2,3,4,3],'Balanced (CHC-style)':[1]*9}
for k,w in P.items():
    w=np.array(w,float);res=[]
    for j in (1,2,3):
        s=np.array([d[j] for d in DIMS],float)
        A=(w*s).sum()/w.sum();G=np.exp((w/w.sum()*np.log(np.maximum(s,1))).sum());res.append((round(A,1),round(G,1)))
    print(k,res)
zs=lambda iq:(iq-100)/15; za=lambda s:norm.ppf(s/200)
for n,iq,s in [("Opus5",129,30.2),("Sol",130,7.8),("Astra est",133,62.7),("Grok4.6 est",127,2.1),("Gemini3.1 Mar",127,0.37)]:
    print(n,round(zs(iq),2),round(za(s),2), [round(100+15*((1-l)*zs(iq)+l*za(s)),1) for l in (0,.25,.5,.75,1)])
for iq in (151,145,135,130): print(iq, round(1/norm.sf(zs(iq))))
# landscape win share
sig=lambda x:1/(1+np.exp(-x))
def Pf(n,h,H,k,b,n0=.2): return sig(k*np.log2(H/h)-b*(n-n0))
import datetime as dt
T0=dt.date(2026,5,8)
def H(d): return 16*2**((d-T0).days/131)
for d in [dt.date(2026,9,15),dt.date(2027,9,15),dt.date(2028,9,15)]:
    n=np.linspace(0,1,21);y=np.linspace(np.log2(1/60),np.log2(2000),25);N,Y=np.meshgrid(n,y);h=2**Y
    gap=Pf(N,h,2000,.5,1.5)-Pf(N,h,H(d),.574,6.0);print(d,round(H(d),1),round((gap>0.25).mean()*100))
for h in (40,167,2000):
    for td in (100,131,200): print(h,td,T0+dt.timedelta(days=np.log2(h/16)*td))
print('k',np.log(4)/np.log2(16/3))

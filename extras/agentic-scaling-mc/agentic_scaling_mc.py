"""
Agentic Scaling Model + Monte Carlo
S. Gondane (2026). Compound AI systems: when does a verified swarm of small models beat one frontier model at equal cost?

Per-step success of an ensemble of n agents (base success p, pairwise error correlation rho),
filtered by a verifier that picks a correct candidate with accuracy v:
    n_eff      = n / (1 + (n-1) rho)                         # correlated samples add less
    P_step(n)  = v * [1 - (1-p)^n_eff]                       # coverage x selection
Chained task of m steps with heterogeneous difficulty d_j:
    P_task     = prod_j P_step(p * d_j)                      # weak-link product
    W          = log P_min / log P_task                      # weak-link share of total failure
Cost at budget B (units of small-model calls):
    n          = floor(B / (m (c_s + c_v)))
Throughput under concurrency (Universal Scalability Law):
    X(N)       = N / (1 + sigma (N-1) + kappa N (N-1)),      N* = sqrt((1-sigma)/kappa)
Self-improvement (training on verified successes, STaR/RL-style):
    logit p_{k+1} = logit p_k + eta * P_step,k * (1 - rho)
"""
import numpy as np, json
rng=np.random.default_rng(42); N=100_000
logit=lambda p:np.log(p/(1-p)); sig=lambda x:1/(1+np.exp(-x))
# priors
p_s=rng.beta(17,3,N)            # small model per-step success ~0.85
p_L=rng.beta(48,2,N)            # frontier model per-step success ~0.96
rho=rng.uniform(0.1,0.6,N)      # error correlation between samples of one small model
v=rng.uniform(0.70,0.98,N)      # verifier selection accuracy
m=rng.integers(5,41,N)          # steps in the task
cost_ratio=rng.uniform(8,30,N)  # frontier call cost / small call cost
c_v=0.3                         # verifier cost per candidate (small units)
eta=rng.uniform(0.0,0.3,N)      # self-improvement rate per round
sigma=rng.uniform(0.01,0.1,N); kappa=rng.uniform(1e-4,5e-3,N)
K=40                             # max steps modelled per task
d=rng.beta(30,1,(N,K))           # step difficulty multipliers (mean ~0.97; a few hard steps)
mask=np.arange(K)[None,:]<m[:,None]

def step(p,n,rho,v):
    neff=n/(1+(n-1)*rho); return v*(1-(1-p)**neff)
def task(pmat):
    lp=np.where(mask,np.log(np.clip(pmat,1e-12,1)),0.0); LP=lp.sum(1)
    Wk=np.where(mask,lp,0).min(1)/np.minimum(LP,-1e-12)
    return np.exp(LP),Wk
# frontier: one call per step, no verifier (baseline), budget = m * cost_ratio small units
PL,WL=task(np.clip(p_L[:,None]*d,0,1))
n=np.maximum(1,np.floor(cost_ratio/(1+c_v))).astype(int)     # same per-step budget spent on swarm
PS,WS=task(step(np.clip(p_s[:,None]*d,0,1),n[:,None],rho[:,None],v[:,None]))
# self-improvement rounds for swarm
win0=PS>PL
p_k=p_s.copy(); rounds=np.full(N,np.inf)
for k in range(1,21):
    Pst=step(p_k,n,rho,v); p_k=sig(logit(np.clip(p_k,1e-6,1-1e-6))+eta*Pst*(1-rho))
    Pk,_=task(step(np.clip(p_k[:,None]*d,0,1),n[:,None],rho[:,None],v[:,None]))
    hit=(Pk>PL)&np.isinf(rounds)&(~win0); rounds[hit]=k
Nstar=np.sqrt((1-sigma)/kappa)
win=PS>PL
res=dict(P_swarm_beats_frontier=float(win.mean()),
 median_P_frontier=float(np.median(PL)),median_P_swarm=float(np.median(PS)),
 median_n=float(np.median(n)),weak_link_share_frontier=float(np.median(WL)),weak_link_share_swarm=float(np.median(WS)),
 P_losers_catch_up_10_rounds=float((rounds[~win0]<=10).mean()),median_rounds_to_catch_up=float(np.median(rounds[np.isfinite(rounds)])),
 median_Nstar=float(np.median(Nstar)))
# sensitivity: SRC of logit gap on parameters
gap=logit(np.clip(PS,1e-9,1-1e-9))-logit(np.clip(PL,1e-9,1-1e-9))
X=np.c_[p_s,p_L,rho,v,m,cost_ratio]; names=['small p','frontier p','error corr rho','verifier v','steps m','cost ratio']
Z=(X-X.mean(0))/X.std(0); b=np.linalg.lstsq(np.c_[np.ones(N),Z],(gap-gap.mean())/gap.std(),rcond=None)[0][1:]
res['sensitivity_SRC']=dict(zip(names,[round(float(x),3) for x in b]))
# win probability by rho and v bins
for lo,hi in [(0.1,0.25),(0.25,0.4),(0.4,0.6)]:
    s=(rho>=lo)&(rho<hi); res[f'win_rho_{lo}-{hi}']=float(win[s].mean())
for lo,hi in [(0.70,0.80),(0.80,0.90),(0.90,0.98)]:
    s=(v>=lo)&(v<hi); res[f'win_v_{lo}-{hi}']=float(win[s].mean())
print(json.dumps(res,indent=1))
np.savez('mc.npz',rho=rho,v=v,win=win,gap=gap,m=m,PS=PS,PL=PL,b=b,rounds=rounds)

# --- Extension: verifier is the ceiling. Condition and co-improving verifier ---
res2={}
res2['P_win_given_v_gt_pL']=float(win[v>p_L].mean()); res2['P_win_given_v_le_pL']=float(win[v<=p_L].mean())
res2['share_v_gt_pL']=float((v>p_L).mean())
# co-improvement: verifier also learns from verified traces: logit v += eta * P_step * (1-rho)
p_k=p_s.copy(); v_k=v.copy(); cur=[]
for k in range(1,21):
    Pst=step(p_k,n,rho,v_k)
    p_k=sig(logit(np.clip(p_k,1e-6,1-1e-6))+eta*Pst*(1-rho))
    v_k=sig(logit(np.clip(v_k,1e-6,1-1e-6))+eta*Pst*(1-rho))
    Pk,_=task(step(np.clip(p_k[:,None]*d,0,1),n[:,None],rho[:,None],v_k[:,None]))
    cur.append(float((Pk>PL).mean()))
res2['win_by_round_coimprove']=[round(x,3) for x in cur]
print(json.dumps(res2,indent=1))
np.save('cur.npy',np.array(cur))

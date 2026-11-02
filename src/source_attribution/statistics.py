from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

def bootstrap_ci(values,n_boot=10000,confidence=.95,seed=42):
    v=np.asarray(values,float); rng=np.random.default_rng(seed)
    means=np.array([rng.choice(v,size=len(v),replace=True).mean() for _ in range(n_boot)])
    a=1-confidence; lo,hi=np.quantile(means,[a/2,1-a/2]); return float(v.mean()),float(lo),float(hi)

def paired_bootstrap_macro_f1(y_true,pred_a,pred_b,n_boot=10000,seed=42):
    y=np.asarray(y_true); a=np.asarray(pred_a); b=np.asarray(pred_b); n=len(y); rng=np.random.default_rng(seed)
    obs=f1_score(y,a,average="macro")-f1_score(y,b,average="macro"); diffs=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,size=n); diffs.append(f1_score(y[idx],a[idx],average="macro")-f1_score(y[idx],b[idx],average="macro"))
    lo,hi=np.quantile(diffs,[.025,.975]); p=2*min(np.mean(np.asarray(diffs)<=0),np.mean(np.asarray(diffs)>=0))
    return {"delta_macro_f1":float(obs),"ci95":[float(lo),float(hi)],"p_value":float(min(1,p))}

def holm_correction(p_values):
    p=np.asarray(p_values,float); order=np.argsort(p); out=np.empty_like(p); running=0.; m=len(p)
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]); out[idx]=min(1.,running)
    return out

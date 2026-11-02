from __future__ import annotations
import numpy as np
from scipy.special import logsumexp,softmax

def msp_id_score(proba): return np.max(np.asarray(proba),axis=1)
def energy_id_score(logits,temperature=1.0):
    T=float(temperature); return T*logsumexp(np.asarray(logits,dtype=float)/T,axis=1)
def sme_id_score(logits,temperature=1.0):
    T=float(temperature); probs=softmax(np.asarray(logits,dtype=float)/T,axis=1)
    return T*logsumexp(probs,axis=1)
def fit_known_quantile_threshold(scores,reject_quantile=0.05):
    s=np.asarray(scores,dtype=float); s=s[np.isfinite(s)]
    if not len(s): raise ValueError("No finite validation ID scores.")
    return float(np.quantile(s,reject_quantile))
def class_prototypes(X,y):
    labels=np.array(sorted(set(map(str,y))),dtype=object); X=np.asarray(X,dtype=float); centers=[]
    for lab in labels:
        c=X[np.asarray(y).astype(str)==lab].mean(axis=0); n=np.linalg.norm(c); centers.append(c/n if n else c)
    return labels,np.asarray(centers)
def prototype_scores(X,centers):
    X=np.asarray(X,dtype=float); n=np.linalg.norm(X,axis=1,keepdims=True); X=X/np.where(n==0,1,n)
    sim=X@centers.T; idx=np.argmax(sim,axis=1); return idx,sim[np.arange(len(X)),idx],sim

from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support,roc_auc_score,average_precision_score,roc_curve

def top_k_accuracy(y_true,proba,classes,k=3):
    classes=np.asarray(classes); top=np.argsort(proba,axis=1)[:,-k:]
    return float(np.mean([str(y) in set(classes[idx].astype(str)) for y,idx in zip(y_true,top)]))

def closed_set_metrics(y_true,y_pred,proba=None,classes=None):
    out={"accuracy":float(accuracy_score(y_true,y_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
         "macro_f1":float(f1_score(y_true,y_pred,average="macro")),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted"))}
    if proba is not None and classes is not None:
        out["top1_accuracy"]=top_k_accuracy(y_true,proba,classes,1); out["top3_accuracy"]=top_k_accuracy(y_true,proba,classes,min(3,len(classes)))
    return out

def per_class_metrics(y_true,y_pred):
    labels=sorted(set(map(str,y_true))|set(map(str,y_pred)))
    p,r,f,s=precision_recall_fscore_support(np.asarray(y_true).astype(str),np.asarray(y_pred).astype(str),labels=labels,zero_division=0)
    return [{"class":lab,"precision":float(pp),"recall":float(rr),"f1":float(ff),"support":int(ss)} for lab,pp,rr,ff,ss in zip(labels,p,r,f,s)]

def fpr95(y_is_id,id_score):
    fpr,tpr,_=roc_curve(np.asarray(y_is_id).astype(int),np.asarray(id_score,dtype=float),pos_label=1)
    idx=np.flatnonzero(tpr>=.95); return float(fpr[idx[0]]) if len(idx) else 1.

def oscr(y_known_true,y_known_pred,known_id_score,unknown_id_score):
    yk=np.asarray(y_known_true).astype(str); pk=np.asarray(y_known_pred).astype(str)
    sk=np.asarray(known_id_score,float); su=np.asarray(unknown_id_score,float); correct=(yk==pk).astype(float)
    thr=np.r_[np.inf,np.unique(np.r_[sk,su])[::-1],-np.inf]; ccr=[]; fpr=[]
    for t in thr:
        ccr.append(float(np.mean(correct*(sk>=t)))); fpr.append(float(np.mean(su>=t)))
    order=np.argsort(fpr)
    return float(np.trapezoid(np.asarray(ccr)[order],np.asarray(fpr)[order]))

def open_set_metrics(
    known_true,
    known_pred,
    known_id_score,
    unknown_id_score,
):
    # All scores use the same convention:
    # larger score = more likely in-distribution / known.
    y_id = np.r_[
        np.ones(len(known_id_score), dtype=int),
        np.zeros(len(unknown_id_score), dtype=int),
    ]

    id_score = np.r_[
        known_id_score,
        unknown_id_score,
    ].astype(float)

    # For AUPR-OOD the positive class must be OOD.
    # Since id_score is larger for ID, negate it to obtain
    # a score that is larger for OOD.
    y_ood = 1 - y_id
    ood_score = -id_score

    return {
        "auroc_id_vs_ood": float(
            roc_auc_score(y_id, id_score)
        ),
        "aupr_ood": float(
            average_precision_score(
                y_ood,
                ood_score,
            )
        ),
        # Keep this only as an auxiliary diagnostic.
        "aupr_id": float(
            average_precision_score(
                y_id,
                id_score,
            )
        ),
        "fpr95": fpr95(
            y_id,
            id_score,
        ),
        "oscr": oscr(
            known_true,
            known_pred,
            known_id_score,
            unknown_id_score,
        ),
    }
#!/usr/bin/env python3
"""Strict paired test-set audit: AASIST vs 4s RuASD baselines.

No training, hyperparameter search, test threshold optimization or model selection.
Only compares previously frozen test predictions for 34 known generators.
Inference: stratified paired bootstrap over *recordings* for fixed fitted models;
McNemar exact test on paired top-1 correctness, Holm adjustment across
four predeclared contrasts (AASIST vs Acoustic LR, WavLM LR, fusion, LightGBM).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support

DEFAULT_FILES = {
    'aasist': 'results/raw/aasist_fixed4_ruasd34_seed42/test_predictions.parquet',
    'acoustic_lr': 'results/raw/fixed4_acoustic_vs_wavlm_lr/acoustic_fixed4/predictions.parquet',
    'wavlm_lr': 'results/raw/fixed4_acoustic_vs_wavlm_lr/wavlm_fixed4/predictions.parquet',
    'fusion': 'results/raw/fixed4_acoustic_wavlm_late_fusion_saved/predictions.parquet',
    'acoustic_lightgbm': 'results/raw/fixed4_acoustic_vs_wavlm_lr/acoustic_fixed4_lightgbm/predictions.parquet',
}
AASIST_SHA256 = 'f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0'


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def load_aligned(splits: Path, paths: dict[str, Path]):
    frozen = pd.read_csv(splits, usecols=['file_id','generator','split'], encoding='utf-8-sig')
    assert not frozen.file_id.astype(str).duplicated().any(), 'Duplicate frozen file_id'
    test = frozen.loc[frozen.split == 'test',['file_id','generator']].copy()
    test.file_id=test.file_id.astype(str); test.generator=test.generator.astype(str)
    assert len(test)==13600 and test.generator.nunique()==34 and (test.generator.value_counts()==400).all(), 'Unexpected RuASD frozen test'
    out = test.rename(columns={'generator':'truth'}).reset_index(drop=True)
    names = set(out.truth)
    for model,path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{model}: missing {path}')
        d = pd.read_parquet(path,columns=['file_id','generator_true','generator_pred']).copy()
        for col in d.columns: d[col] = d[col].astype(str)
        if len(d)!=13600 or d.file_id.duplicated().any() or set(d.file_id)!=set(out.file_id):
            raise ValueError(f'{model}: test file_id alignment/uniqueness violation')
        if not set(d.generator_pred).issubset(names):
            raise ValueError(f'{model}: unexpected predicted classes')
        d = d.rename(columns={'generator_true':f'{model}_truth','generator_pred':model})
        out = out.merge(d, on='file_id',validate='one_to_one',how='left',sort=False)
        if not (out.truth == out[f'{model}_truth']).all():
            raise ValueError(f'{model}: true labels disagree with frozen split')
        out.drop(columns=[f'{model}_truth'], inplace=True)
    return out, sorted(names)


def check_aasist_metadata(frozen: Path, report: Path, aligned: pd.DataFrame, expected_labels: list[str]):
    if not report.is_file(): raise FileNotFoundError(f'AASIST final metrics missing: {report}')
    j=json.loads(report.read_text(encoding='utf8'))
    p=j['protocol']
    actual_hash=file_hash(frozen)
    if actual_hash != p['frozen_split_sha256'] or actual_hash != AASIST_SHA256:
        raise ValueError('Frozen split SHA256 differs from trained AASIST model')
    if set(p['classes_in_output_order']) != set(expected_labels) or p['n_classes']!=34:
        raise ValueError('AASIST classes mismatch')
    if p['seed']!=42 or j['checkpoint_epoch']!=16:
        raise ValueError('AASIST is not the seed42 validated epoch16 result')
    baseline=j['test']
    for metric,value in [('macro_f1',f1_score(aligned.truth,aligned.aasist,labels=expected_labels,average='macro',zero_division=0)),
                         ('accuracy',accuracy_score(aligned.truth,aligned.aasist)),
                         ('balanced_accuracy',balanced_accuracy_score(aligned.truth,aligned.aasist))]:
        if not np.isclose(float(baseline[metric]),value,atol=1e-12,rtol=0):
            raise ValueError(f'AASIST reconstructed {metric} differs from official report')
    return j,actual_hash


def f1_confusion(cm):
    tp=np.diag(cm).astype(float)
    denom=cm.sum(axis=0)+cm.sum(axis=1)
    return float(np.mean(np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom!=0)))


def holm_adjust(pvalues):
    out=np.zeros(len(pvalues),dtype=float)
    level=0.0
    for r,i in enumerate(np.argsort(pvalues)):
        level=max(level, min(1.0,(len(pvalues)-r)*float(pvalues[i])))
        out[i]=level
    return out.tolist()


def score_pairs(aligned:pd.DataFrame, classes:list[str],n_bootstrap:int,seed:int):
    if n_bootstrap<100:raise ValueError('Use at least 100 bootstrap iterations')
    methods=[x for x in DEFAULT_FILES if x in aligned.columns]
    classes_idx={name:i for i,name in enumerate(classes)}
    y=np.array([classes_idx[v] for v in aligned.truth],dtype=np.int16)
    p={m:np.array([classes_idx[v] for v in aligned[m]],dtype=np.int16) for m in methods}
    n=len(y); k=len(classes);rng=np.random.default_rng(seed)
    strata=[np.flatnonzero(y==i) for i in range(k)]
    dist={m:{'macro_f1':np.empty(n_bootstrap), 'accuracy':np.empty(n_bootstrap)} for m in methods}
    for b in range(n_bootstrap):
        idx=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
        for m in methods:
            cm=np.bincount(y[idx].astype(np.int32)*k+p[m][idx],minlength=k*k).reshape(k,k)
            dist[m]['macro_f1'][b]=f1_confusion(cm)
            dist[m]['accuracy'][b]=np.trace(cm)/n
    ci=lambda x:[float(v) for v in np.quantile(x,[.025,.975])]
    metrics={}
    for m in methods:
        metrics[m]={'accuracy':float(accuracy_score(y,p[m])),
                    'balanced_accuracy':float(balanced_accuracy_score(y,p[m])),
                    'macro_f1':float(f1_score(y,p[m],labels=np.arange(k),average='macro',zero_division=0)),
                    'accuracy_95ci':ci(dist[m]['accuracy']),
                    'macro_f1_95ci':ci(dist[m]['macro_f1'])}
    pairs=[]
    for other in ('acoustic_lr','wavlm_lr','fusion','acoustic_lightgbm'):
        if other not in p:continue
        a=p['aasist']==y; z=p[other]==y
        only_a=int((a&(~z)).sum()); only_b=int(((~a)&z).sum()); discordant=only_a+only_b
        mcp=float(binomtest(only_a,discordant,p=.5).pvalue) if discordant else 1.0
        record={'first':'aasist','second':other,'only_first_correct':only_a,
                'only_second_correct':only_b,'mcnemar_exact_p':mcp}
        for key in ('macro_f1','accuracy'):
            record[f'delta_{key}']=metrics['aasist'][key]-metrics[other][key]
            record[f'delta_{key}_95ci']=ci(dist['aasist'][key]-dist[other][key])
        pairs.append(record)
    pvals=holm_adjust([a['mcnemar_exact_p'] for a in pairs])
    for row,pval in zip(pairs,pvals):row['mcnemar_holm_p']=pval
    # Class-level differences, no selection by performance.
    perclass=[]
    for other in ('acoustic_lr','wavlm_lr','fusion','acoustic_lightgbm'):
        if other not in p:continue
        for c,lab in enumerate(classes):
            mask=(y==c)
            rec_a=float((p['aasist'][mask]==y[mask]).mean())
            rec_b=float((p[other][mask]==y[mask]).mean())
            perclass.append({'comparison':f'aasist_minus_{other}', 'generator':lab,
                             'aasist_recall':rec_a,'baseline_recall':rec_b,
                             'delta_recall':rec_a-rec_b,'n_test':int(mask.sum())})
    return metrics,pairs,perclass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits',default='splits/closed_set.csv')
    for model,default in DEFAULT_FILES.items():
        ap.add_argument('--'+model.replace('_','-'), default=default)
    ap.add_argument('--report',default='results/raw/aasist_fixed4_ruasd34_seed42/final_metrics.json')
    ap.add_argument('--out',default='results/statistical_tests/fixed4_aasist_vs_baselines_seed42')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    inputs={m:Path(getattr(args,m)) for m in DEFAULT_FILES}
    # optional fixed4 LightGBM if its saved prediction location differs/wasn't retained
    if not inputs['acoustic_lightgbm'].exists():
        print('INFO: LightGBM fixed4 predictions not found at default path; skipping ONLY that optional comparison.')
        inputs.pop('acoustic_lightgbm')
    aligned,classes=load_aligned(Path(args.splits),inputs)
    report,split_sha=check_aasist_metadata(Path(args.splits),Path(args.report),aligned,classes)
    metrics,pairs,perclass=score_pairs(aligned,classes,args.bootstrap,args.seed)
    out=Path(args.out)
    if out.exists():raise FileExistsError('Refusing to overwrite: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    doc={'protocol':'RuASD_34class_first4s_AASIST_paired_prediction_test_audit',
         'aasist_seed':42,'aasist_best_epoch':report['checkpoint_epoch'],
         'frozen_split_sha256':split_sha,'n_test':len(aligned),'n_classes':len(classes),
         'bootstrap':{'repetitions':args.bootstrap,'seed':args.seed,
           'stratification':'generator_true, 400 recordings per class, same indices for all models',
           'scope':'recording-level uncertainty for fixed trained models; not repeat-training, new speakers, or other corpora'},
         'holm_family':'All predeclared AASIST-vs-other-model comparisons present in the output',
         'input_paths':{m:str(p) for m,p in inputs.items()},
         'model_metrics':metrics,'paired_comparisons':pairs}
    (out/'metrics.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(pairs).to_csv(out/'paired_comparisons.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(perclass).to_csv(out/'per_class_recall_differences.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(doc,ensure_ascii=False,indent=2))
    print('\nOUTPUT:',out)

if __name__=='__main__':main()

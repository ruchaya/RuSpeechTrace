from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,yaml
from sklearn.metrics import confusion_matrix
from source_attribution.experiment import feature_columns
from source_attribution.models import build_and_train
from source_attribution.metrics import closed_set_metrics

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    rp=Path('splits/closed_set.csv');mp=Path('splits/mlaad_ru_three_family_split.csv')
    rf=Path('features/ruasd_closed_set_acoustic.parquet');mf=Path('features/mlaad_three_family_acoustic.parquet')
    mapping_file=Path('data_inventory/family_mapping_ruasd_mlaad_three_family_candidate.csv')
    out=Path('results/raw/two_family_transfer_seed42')
    assert not out.exists(),'RESULT DIRECTORY ALREADY EXISTS'
    mapping=pd.read_csv(mapping_file,encoding='utf-8-sig')
    mapping=mapping[mapping.family_label.ne('MMS-TTS')].copy()
    assert len(mapping)==2 and mapping['family_label'].is_unique
    labels=sorted(mapping['family_label'].tolist())
    rmeta=json.loads(rf.with_suffix('.meta.json').read_text(encoding='utf-8'))
    mmeta=json.loads(mf.with_suffix('.meta.json').read_text(encoding='utf-8'))
    assert rmeta['feature_signature']==mmeta['feature_signature']
    assert rmeta['split_sha256']==sha(rp) and mmeta['split_sha256']==sha(mp)
    rs=pd.read_csv(rp,usecols=['file_id','split','generator'])
    rfeatures=pd.read_parquet(rf).drop(columns=['generator','split'],errors='ignore')
    assert rs['file_id'].is_unique and rfeatures['file_id'].is_unique
    r=rs.merge(rfeatures,on='file_id',how='inner',validate='one_to_one')
    assert len(r)==68000
    m=pd.read_parquet(mf)
    frozen=pd.read_csv(mp,encoding='utf-8-sig',usecols=['path','split','generator'])
    check=m[['path','split','generator']].merge(frozen,on='path',validate='one_to_one',suffixes=('_f','_s'))
    assert len(check)==len(m)==3000
    assert (check['split_f']==check['split_s']).all() and (check['generator_f']==check['generator_s']).all()
    r['family_target']=r['generator'].map(dict(zip(mapping['ruasd_model'],mapping['family_label'])))
    m['family_target']=m['generator'].map(dict(zip(mapping['mlaad_model'],mapping['family_label'])))
    r=r.loc[r['family_target'].notna()].copy();m=m.loc[m['family_target'].notna()].copy()
    assert len(r)==4000 and len(m)==2000
    for d,expected in [(r,[1300,300,400]),(m,[650,150,200])]:
        tab=pd.crosstab(d['family_target'],d['split']).reindex(index=labels,columns=['train','validation','test'],fill_value=0)
        assert (tab.to_numpy()==np.array(expected)).all(),str(tab)
    cols=feature_columns(r,['spectral_','prosody_'],exclude_duration=True)
    assert len(cols)==789 and cols==feature_columns(m,['spectral_','prosody_'],exclude_duration=True)
    assert 'prosody_duration_sec' not in cols
    for d in (r,m):assert not np.isinf(d[cols].to_numpy(dtype=float)).any()
    model_cfg=yaml.safe_load(Path('configs/cross_dataset.yaml').read_text(encoding='utf-8'))['model']
    scores={};predictions=[]
    for source,trainset in [('ruasd',r),('mlaad',m)]:
        tr=trainset.loc[trainset['split']=='train'];va=trainset.loc[trainset['split']=='validation']
        model=build_and_train(tr[cols],tr['family_target'],va[cols],va['family_target'],model_cfg['type'],model_cfg['params'],80,42)
        assert sorted(model.classes_.tolist())==labels
        for dest,evalset in [('ruasd',r),('mlaad',m)]:
            te=evalset.loc[evalset['split']=='test'];tag=source+'_to_'+dest
            y=te['family_target'].astype(str).to_numpy();pred=model.predict(te[cols]);prob=model.predict_proba(te[cols])
            scores[tag]={'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'metrics':closed_set_metrics(y,pred,prob,model.classes_),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}
            key=te['file_id'] if dest=='ruasd' else te['path']
            q=pd.DataFrame({'experiment':tag,'sample_key':key.to_numpy(),'true_family':y,'predicted_family':pred})
            for j,c in enumerate(model.classes_):q['proba_'+c]=prob[:,j]
            predictions.append(q)
            print(tag,'N:',len(te),'MACRO_F1:',round(scores[tag]['metrics']['macro_f1'],4),'BAL_ACC:',round(scores[tag]['metrics']['balanced_accuracy'],4),flush=True)
    out.mkdir(parents=True)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    report={'seed':42,'family_labels':labels,'features':cols,'mapping_sha256':sha(mapping_file),'ruasd_split_sha256':sha(rp),'mlaad_split_sha256':sha(mp),'model_config':model_cfg,'results':scores}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED:',out,'ALL FOUR EXPERIMENTS PASSED')

if __name__=='__main__':main()

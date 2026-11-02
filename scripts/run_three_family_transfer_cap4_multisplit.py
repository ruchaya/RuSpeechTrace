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
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True,choices=[42,123,456,789,2026])
    seed=parser.parse_args().seed
    hashes={42: '2a40af3150b39b0d0f52f1d09a33a668f0dd6fb37aee86f45f4c28923cae4971', 123: '1ecd5468aa2cf967e4b1189fa22ab2ef9ef5b54f3d9a3bb94c4d189d275da509', 456: 'b65c672a5c1447d20d3f53a96949627d9a901bf5d9fa7ea17cbd79061f51e168', 789: '39bf6ac21d2a39e481372a55ec6704f619f918cb611134a395cc385b28941fcb', 2026: '60d20e2fd5c2a6c7494f86ff887d26942b2daf1e8a38897407f0c2fdd6348d5e'}
    rp=Path('splits/closed_set.csv');mp=Path('splits/mlaad_ru_three_family_split_v2.csv' if seed==2026 else f'splits/mlaad_ru_three_family_split_seed{seed}.csv')
    rf=Path('features/ruasd_closed_set_acoustic_fixed4.parquet');mf=Path('features/mlaad_three_family_acoustic_cap4.parquet')
    mapping_file=Path('data_inventory/family_mapping_ruasd_mlaad_three_family_candidate.csv')
    out=Path(f'results/raw/three_family_transfer_cap4_multisplit_seed{seed}')
    assert not out.exists(),'RESULT DIRECTORY ALREADY EXISTS'
    mapping=pd.read_csv(mapping_file,encoding='utf-8-sig')
    assert len(mapping)==3 and mapping['family_label'].is_unique
    labels=sorted(mapping['family_label'].tolist())
    rmeta=json.loads(rf.with_suffix('.meta.json').read_text(encoding='utf-8'))
    mmeta=json.loads(mf.with_suffix('.meta.json').read_text(encoding='utf-8'))
    assert rmeta['feature_signature']==mmeta['feature_signature']
    assert rmeta['split_sha256']==sha(rp) and mmeta['split_sha256']==sha(Path('splits/mlaad_ru_three_family_split.csv'))
    assert sha(mp)==hashes[seed]
    rs=pd.read_csv(rp,usecols=['file_id','split','generator'])
    rfeatures=pd.read_parquet(rf).drop(columns=['generator','split'],errors='ignore')
    assert rs['file_id'].is_unique and rfeatures['file_id'].is_unique
    r=rs.merge(rfeatures,on='file_id',how='inner',validate='one_to_one')
    assert len(r)==68000
    m=pd.read_parquet(mf)
    frozen=pd.read_csv(mp,encoding='utf-8-sig',usecols=['path','split','split_v1','generator'])
    check=m[['path','split','generator']].merge(frozen,on='path',validate='one_to_one',suffixes=('_f','_s'))
    assert len(check)==len(m)==3000
    assert (check['split_f']==check['split_v1']).all() and (check['generator_f']==check['generator_s']).all()
    assert frozen['path'].is_unique
    m['split']=m['path'].map(frozen.set_index('path')['split'])
    assert m['split'].notna().all()
    r['family_target']=r['generator'].map(dict(zip(mapping['ruasd_model'],mapping['family_label'])))
    m['family_target']=m['generator'].map(dict(zip(mapping['mlaad_model'],mapping['family_label'])))
    r=r.loc[r['family_target'].notna()].copy();m=m.loc[m['family_target'].notna()].copy()
    assert len(r)==6000 and len(m)==3000
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
        model=build_and_train(tr[cols],tr['family_target'],va[cols],va['family_target'],model_cfg['type'],model_cfg['params'],80,seed)
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
    report={'seed':seed,'family_labels':labels,'features':cols,'mapping_sha256':sha(mapping_file),'ruasd_split_sha256':sha(rp),'mlaad_split_sha256':sha(mp),'model_config':model_cfg,'results':scores}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED:',out,'ALL FOUR EXPERIMENTS PASSED')

if __name__=='__main__':main()

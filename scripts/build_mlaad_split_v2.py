import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

old=Path('splits/mlaad_ru_three_family_split.csv')
out=Path('splits/mlaad_ru_three_family_split_v2.csv')
assert old.is_file() and not out.exists()
d=pd.read_csv(old,encoding='utf-8-sig',keep_default_na=False).sort_values('path').reset_index(drop=True)
assert len(d)==3000 and d['path'].is_unique
families=['Bark','MMS-TTS','XTTS-v2']
parts=['train','validation','test']
quota=np.array([650,150,200])
sizes=d.groupby('component').size().sort_index()
ids=sizes[sizes.gt(1)].index
assert len(ids)==142 and sizes.loc[ids].sum()==392
multi=d[d.component.isin(ids)]
M=pd.crosstab(multi.component,multi.family).reindex(index=ids,columns=families,fill_value=0).to_numpy()
probs=quota/1000
expected=M.sum(axis=0)[:,None]*probs[None,:]
rng=np.random.default_rng(2026)
best=None
score_best=float('inf')
for trial in range(1024):
 a=rng.choice(3,size=len(ids),p=probs)
 counts=np.stack([M[a==j].sum(axis=0) for j in range(3)],axis=1)
 if (counts>quota[None,:]).any() or (np.bincount(a,minlength=3)==0).any():continue
 score=float(np.square(counts-expected).sum())
 if score<score_best:best=a.copy();score_best=score
assert best is not None
d['split_v2']=d.component.map(dict(zip(ids,[parts[j] for j in best])))
for family in families:
 mask=d.family.eq(family)
 singles=d.index[mask & d.split_v2.isna()].to_numpy()
 used=d.loc[mask,'split_v2'].value_counts().reindex(parts,fill_value=0).to_numpy()
 need=quota-used
 assert (need>=0).all() and need.sum()==len(singles)
 shuffled=rng.permutation(singles)
 start=0
 for part,n in zip(parts,need):
  d.loc[shuffled[start:start+int(n)],'split_v2']=part
  start+=int(n)
assert d.split_v2.notna().all()
for key in ['component','text_key','original_file']:
 assert d.groupby(key)['split_v2'].nunique().max()==1,key
balance=pd.crosstab(d.family,d.split_v2).reindex(index=families,columns=parts,fill_value=0)
assert (balance.to_numpy()==np.tile(quota,(3,1))).all()
g=d.groupby('component').agg(size=('path','size'),part=('split_v2','first'))
multis=g.loc[g['size'].gt(1),'part'].value_counts().reindex(parts,fill_value=0)
assert (multis>0).all()
d=d.rename(columns={'split':'split_v1','split_v2':'split'})
d.to_csv(out,index=False,encoding='utf-8-sig')
print('SAVED:',out)
print('CLASS BALANCE:')
print(balance.to_string())
print('MULTI COMPONENTS:')
print(multis.to_string())
print('SHA256:',hashlib.sha256(out.read_bytes()).hexdigest())
print('LEAKAGE CHECK PASSED')

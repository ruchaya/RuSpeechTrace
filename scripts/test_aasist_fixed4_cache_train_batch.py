#!/usr/bin/env python
"""One cached train update + validation read, no test and no saved checkpoint."""
import os
import argparse
from pathlib import Path
import torch
from run_aasist_fixed4_ruasd import load_split, load_official_aasist, sha256, seed_everything
from aasist_fixed4_cache_utils import Fixed4CacheDataset

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--splits',type=Path,default=Path('splits/closed_set.csv'))
    ap.add_argument('--tar-root',type=Path,default=Path(os.environ.get("RUASD_ROOT", "data/raw/RuASD")))
    ap.add_argument('--cache-root',type=Path,default=Path('data/cache/aasist_fixed4_float32'))
    ap.add_argument('--repo-root',type=Path,default=Path('third_party/aasist'))
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA')
    seed_everything(42)
    df,labels=load_split(args.splits,args.tar_root)
    digest=sha256(args.splits)
    ds=Fixed4CacheDataset(df[df.split=='train'],labels,args.cache_root,'train',digest)
    vs=Fixed4CacheDataset(df[df.split=='validation'],labels,args.cache_root,'validation',digest)
    try:
        xb=torch.stack([ds[i][0] for i in range(4)],dim=0).cuda()
        yb=torch.tensor([ds[i][1] for i in range(4)],device='cuda')
        assert xb.shape==(4,64600)
        assert torch.count_nonzero(xb[:,64000:])==0
        m,cfg=load_official_aasist(args.repo_root,34)
        assert cfg['nb_samp']==64600
        m=m.cuda().train()
        optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-4)
        _,logits=m(xb,Freq_aug=False)
        loss=torch.nn.functional.cross_entropy(logits.float(),yb)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError('Non-finite cached train loss')
        loss.backward()
        optimizer.step()
        vx,vy,vfid=vs[0]
        assert vx.shape==(64600,) and torch.isfinite(vx).all().item()
        print('CACHE GPU PASS: train batch=4 forward/backward, val read, exact 64000+600 input; '
              f'loss={loss.item():.5f}; memory_MiB={torch.cuda.max_memory_allocated()/2**20:.0f}')
        print('No test WAV accessed. No checkpoint saved. Not a performance estimate.')
    finally:
        ds.close()
        vs.close()

if __name__=='__main__':
    main()

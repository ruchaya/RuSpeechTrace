#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import LabelEncoder
from source_attribution.aasist_adapter import load_official_aasist
from source_attribution.audio import load_audio,fixed_length
from source_attribution.metrics import closed_set_metrics
from source_attribution.experiment import create_run_dir
from source_attribution.utils import json_dump,set_seed

class DS(Dataset):
    def __init__(self,df,le,nb):self.df=df.reset_index(drop=True);self.le=le;self.nb=nb
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];y,sr=load_audio(r["path"],16000,peak_normalize=False)
        if len(y)<self.nb:
            y=np.tile(y,int(np.ceil(self.nb/max(1,len(y)))))[:self.nb] if len(y) else np.zeros(self.nb,np.float32)
        else:y=fixed_length(y,sr,self.nb/sr)
        return torch.tensor(y,dtype=torch.float32),torch.tensor(self.le.transform([str(r["generator"])])[0]),str(r["file_id"])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--splits",required=True);ap.add_argument("--repo-root",default="third_party/aasist");ap.add_argument("--epochs",type=int,default=20);ap.add_argument("--batch-size",type=int,default=12);ap.add_argument("--lr",type=float,default=1e-4);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--output-root",default="results/raw");a=ap.parse_args();set_seed(a.seed)
    df=pd.read_csv(a.splits)
    if "path" not in df:raise ValueError("Split CSV must include audio path.")
    le=LabelEncoder().fit(df[df["split"]=="train"]["generator"].astype(str))
    if not set(df[df["split"]=="test"]["generator"].astype(str))<=set(le.classes_):raise RuntimeError("Test has unseen classes.")
    model,cfg=load_official_aasist(a.repo_root,len(le.classes_));dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=model.to(dev);opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-4);lossfn=torch.nn.CrossEntropyLoss()
    loaders={s:DataLoader(DS(df[df["split"]==s],le,cfg["nb_samp"]),batch_size=a.batch_size,shuffle=(s=="train"),num_workers=0) for s in ["train","validation","test"]};best=None;bestloss=float("inf")
    for _ in range(a.epochs):
        model.train()
        for x,y,_ in loaders["train"]:
            opt.zero_grad();_,logits=model(x.to(dev),Freq_aug=False);loss=lossfn(logits,y.to(dev));loss.backward();opt.step()
        model.eval();ls=[]
        with torch.inference_mode():
            for x,y,_ in loaders["validation"]:_h,z=model(x.to(dev),Freq_aug=False);ls.append(lossfn(z,y.to(dev)).item())
        vl=float(np.mean(ls)) if ls else float("inf")
        if vl<bestloss:bestloss=vl;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best is None:raise RuntimeError("No AASIST checkpoint.")
    model.load_state_dict(best);model.eval();true=[];pred=[];prob=[];ids=[]
    with torch.inference_mode():
        for x,y,fid in loaders["test"]:
            _,z=model(x.to(dev),Freq_aug=False);p=torch.softmax(z,1).cpu().numpy();prob.append(p);yy=y.numpy();true.extend(le.inverse_transform(yy));pred.extend(le.inverse_transform(np.argmax(p,1)));ids.extend(fid)
    prob=np.vstack(prob);met=closed_set_metrics(true,pred,prob,le.classes_);run=create_run_dir(a.output_root,"closed_set_aasist",a.seed);json_dump({"metrics":met,"model_config":cfg},run/"metrics.json");pd.DataFrame({"file_id":ids,"generator_true":true,"generator_pred":pred}).to_parquet(run/"predictions.parquet",index=False);torch.save({"state_dict":best,"classes":le.classes_.tolist(),"model_config":cfg},run/"model.pt");print(met);print(run)
if __name__=="__main__":main()

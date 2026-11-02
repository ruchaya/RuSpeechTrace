from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa
from scipy.fft import dct
from scipy.stats import skew,kurtosis
EPS=1e-10

def _stats(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {k:np.nan for k in ["mean","std","min","max","median","skew","kurtosis"]}
    return {"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),"max":float(x.max()),
            "median":float(np.median(x)),"skew":float(skew(x,bias=False)) if len(x)>2 and x.std()>0 else 0.0,
            "kurtosis":float(kurtosis(x,fisher=True,bias=False)) if len(x)>3 and x.std()>0 else 0.0}

def _summarize(mat,prefix):
    mat=np.asarray(mat)
    if mat.ndim==1: mat=mat[None,:]
    out={}
    for i,row in enumerate(mat):
        for k,v in _stats(row).items(): out[f"{prefix}_{i:02d}_{k}"]=v
    return out

def _linear_fb(sr,n_fft,n_filters=40):
    freqs=np.linspace(0,sr/2,n_fft//2+1); edges=np.linspace(0,sr/2,n_filters+2)
    fb=np.zeros((n_filters,len(freqs)))
    for m in range(n_filters):
        l,c,r=edges[m:m+3]
        fb[m]=np.maximum(0,np.minimum((freqs-l)/max(c-l,EPS),(r-freqs)/max(r-c,EPS)))
    return fb

def _lfcc(y,sr,n_coeff=20,n_fft=400,hop=160,n_filters=40):
    S=np.abs(librosa.stft(y,n_fft=n_fft,hop_length=hop,win_length=n_fft,window="hamming"))**2
    E=_linear_fb(sr,n_fft,n_filters)@S
    return dct(np.log(E+EPS),type=2,axis=0,norm="ortho")[:n_coeff]

def _cqcc_approx(y,sr,n_coeff=20):
    fmin=max(32.0,librosa.note_to_hz("C1"))
    C=np.abs(librosa.cqt(y,sr=sr,fmin=fmin,n_bins=72,bins_per_octave=12))
    return dct(np.log(C+EPS),type=2,axis=0,norm="ortho")[:n_coeff]

def _pause(rms,hop,sr):
    if len(rms)==0:return 0.,0.,0.
    silent=rms<=np.quantile(rms,0.25); runs=[]; cur=0
    for s in silent:
        if s: cur+=1
        elif cur: runs.append(cur); cur=0
    if cur:runs.append(cur)
    d=np.asarray(runs,dtype=float)*hop/sr
    return float(len(runs)),float(d.mean()) if len(d) else 0.,float(silent.mean())

@dataclass
class AcousticFeatureConfig:
    sr:int=16000
    n_fft:int=400
    hop:int=160
    n_mfcc:int=20
    include_duration:bool=True

def extract_acoustic_features(y,sr,original_duration=None,cfg=None):
    cfg=cfg or AcousticFeatureConfig(sr=sr); n_fft=cfg.n_fft; hop=cfg.hop
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=cfg.n_mfcc,n_fft=n_fft,hop_length=hop,win_length=n_fft,window="hamming")
    delta=librosa.feature.delta(mfcc,width=9,order=1,mode="nearest")
    delta2=librosa.feature.delta(mfcc,width=9,order=2,mode="nearest")
    lfcc=_lfcc(y,sr,cfg.n_mfcc,n_fft,hop,40); cqcc=_cqcc_approx(y,sr,cfg.n_mfcc)
    out={}; out.update(_summarize(mfcc,"spectral_mfcc")); out.update(_summarize(delta,"spectral_mfcc_delta"))
    out.update(_summarize(delta2,"spectral_mfcc_delta2")); out.update(_summarize(lfcc,"spectral_lfcc")); out.update(_summarize(cqcc,"spectral_cqcc"))
    sequences={
        "centroid":librosa.feature.spectral_centroid(y=y,sr=sr,n_fft=n_fft,hop_length=hop)[0],
        "bandwidth":librosa.feature.spectral_bandwidth(y=y,sr=sr,n_fft=n_fft,hop_length=hop)[0],
        "rolloff":librosa.feature.spectral_rolloff(y=y,sr=sr,n_fft=n_fft,hop_length=hop)[0],
        "flatness":librosa.feature.spectral_flatness(y=y,n_fft=n_fft,hop_length=hop)[0],
    }
    for name,seq in sequences.items():
        for k,v in _stats(seq).items(): out[f"spectral_stft_{name}_{k}"]=v
    contrast=librosa.feature.spectral_contrast(y=y,sr=sr,n_fft=512,hop_length=hop)
    out.update(_summarize(contrast,"spectral_stft_contrast"))
    try:
        f0,voiced,_=librosa.pyin(y,fmin=60,fmax=500,sr=sr,frame_length=2048,hop_length=hop)
    except Exception:
        f0=np.full(max(1,len(y)//hop),np.nan); voiced=np.zeros_like(f0,dtype=bool)
    vf=f0[np.isfinite(f0)]
    out["prosody_f0_mean"]=float(vf.mean()) if len(vf) else np.nan
    out["prosody_f0_std"]=float(vf.std()) if len(vf) else np.nan
    out["prosody_f0_min"]=float(vf.min()) if len(vf) else np.nan
    out["prosody_f0_max"]=float(vf.max()) if len(vf) else np.nan
    out["prosody_voiced_ratio"]=float(np.mean(voiced)) if len(voiced) else 0.
    rms=librosa.feature.rms(y=y,frame_length=2048,hop_length=hop)[0]
    zcr=librosa.feature.zero_crossing_rate(y,frame_length=2048,hop_length=hop)[0]
    out.update({"prosody_rms_mean":float(rms.mean()),"prosody_rms_std":float(rms.std()),
                "prosody_zcr_mean":float(zcr.mean()),"prosody_zcr_std":float(zcr.std())})
    pc,pm,pr=_pause(rms,hop,sr)
    out.update({"prosody_pause_count":pc,"prosody_pause_mean_duration":pm,"prosody_pause_ratio":pr})
    if cfg.include_duration:
        out["prosody_duration_sec"]=float(original_duration) if original_duration is not None and np.isfinite(original_duration) else float(len(y)/sr)
    return out

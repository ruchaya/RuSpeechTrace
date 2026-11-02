from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

def load_audio(path,target_sr=16000,mono=True,peak_normalize=True):
    y,sr=sf.read(str(path),always_2d=False); y=np.asarray(y,dtype=np.float32)
    if y.ndim==2 and mono: y=y.mean(axis=1)
    if sr!=target_sr:
        y=librosa.resample(y,orig_sr=sr,target_sr=target_sr,res_type="kaiser_best"); sr=target_sr
    if len(y): y=y-float(np.mean(y))
    if peak_normalize and len(y):
        peak=float(np.max(np.abs(y)))
        if peak>0: y=y/peak
    return y.astype(np.float32),int(sr)

def fixed_length(y,sr,seconds,mode="center"):
    n=int(round(sr*seconds))
    if len(y)<n: return np.pad(y,(0,n-len(y)))
    if len(y)==n: return y
    if mode=="start": return y[:n]
    start=max(0,(len(y)-n)//2); return y[start:start+n]

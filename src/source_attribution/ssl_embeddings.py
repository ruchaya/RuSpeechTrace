from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from transformers import AutoFeatureExtractor,AutoModel

@dataclass
class SSLConfig:
    model_id:str="microsoft/wavlm-base"
    revision:str|None=None
    sample_rate:int=16000
    pooling:str="mean"
    device:str="auto"

class SSLExtractor:
    def __init__(self,cfg):
        self.cfg=cfg
        self.device=torch.device("cuda" if cfg.device=="auto" and torch.cuda.is_available() else ("cpu" if cfg.device=="auto" else cfg.device))
        self.processor=AutoFeatureExtractor.from_pretrained(cfg.model_id,revision=cfg.revision)
        self.model=AutoModel.from_pretrained(cfg.model_id,revision=cfg.revision).to(self.device).eval()
    @torch.inference_mode()
    def encode_batch(self,audios):
        inputs=self.processor(audios,sampling_rate=self.cfg.sample_rate,return_tensors="pt",padding=True)
        iv=inputs["input_values"].to(self.device)
        am=inputs.get("attention_mask")
        if am is not None: am=am.to(self.device)
        h=self.model(input_values=iv,attention_mask=am).last_hidden_state
        if self.cfg.pooling!="mean": raise ValueError(self.cfg.pooling)
        return h.mean(dim=1).cpu().numpy().astype(np.float32)

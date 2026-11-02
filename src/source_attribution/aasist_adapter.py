from pathlib import Path
import importlib.util
import torch.nn as nn

DEFAULT_MODEL_CONFIG={"architecture":"AASIST","nb_samp":64600,"first_conv":128,"filts":[70,[1,32],[32,32],[32,64],[64,64]],"gat_dims":[64,32],"pool_ratios":[0.5,0.7,0.5,0.5],"temperatures":[2.0,2.0,100.0,100.0]}

def load_official_aasist(repo_root,num_classes,model_config=None):
    model_file=Path(repo_root)/"models"/"AASIST.py"
    if not model_file.exists(): raise FileNotFoundError(f"Run scripts/install_aasist.py; missing {model_file}")
    spec=importlib.util.spec_from_file_location("official_aasist",model_file); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg=dict(DEFAULT_MODEL_CONFIG); cfg.update(model_config or {})
    model=module.Model(cfg)
    if not hasattr(model,"out_layer"): raise AttributeError("AASIST adapter requires official Model.out_layer")
    model.out_layer=nn.Linear(model.out_layer.in_features,num_classes)
    return model,cfg

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

@dataclass
class TrainedClassifier:
    estimator:object
    classes_:np.ndarray
    label_encoder:LabelEncoder
    kind:str
    def predict_proba(self,X): return np.asarray(self.estimator.predict_proba(X))
    def predict(self,X):
        idx=np.argmax(self.predict_proba(X),axis=1); return self.classes_[idx]
    def decision_scores(self,X):
        if self.kind=="lightgbm":
            raw=np.asarray(self.estimator.predict(X,raw_score=True))
            return np.column_stack([-raw,raw]) if raw.ndim==1 else raw
        if hasattr(self.estimator,"decision_function"):
            raw=np.asarray(self.estimator.decision_function(X))
            return np.column_stack([-raw,raw]) if raw.ndim==1 else raw
        return np.log(np.clip(self.predict_proba(X),1e-12,1.))

def build_and_train(X_train,y_train,X_val=None,y_val=None,model_type="lightgbm",params=None,early_stopping_rounds=80,seed=42):
    params=dict(params or {}); le=LabelEncoder(); ytr=le.fit_transform(np.asarray(y_train).astype(str))
    yv=le.transform(np.asarray(y_val).astype(str)) if y_val is not None else None
    if model_type=="lightgbm":
        params.setdefault("objective","multiclass"); params.setdefault("num_class",len(le.classes_)); params.setdefault("random_state",seed); params.setdefault("n_jobs",-1)
        est=lgb.LGBMClassifier(**params); kw={}
        if X_val is not None and y_val is not None:
            kw={"eval_set":[(X_val,yv)],"eval_metric":"multi_logloss"}
            if early_stopping_rounds: kw["callbacks"]=[lgb.early_stopping(early_stopping_rounds,verbose=False)]
        est.fit(X_train,ytr,**kw); return TrainedClassifier(est,le.classes_,le,"lightgbm")
    if model_type=="logistic":
        params.setdefault("max_iter",3000); params.setdefault("class_weight","balanced"); params.setdefault("random_state",seed)
        est=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("classifier",LogisticRegression(**params))])
        est.fit(X_train,ytr); return TrainedClassifier(est,le.classes_,le,"logistic")
    if model_type=="mlp":
        params.setdefault("hidden_layer_sizes",(256,128)); params.setdefault("max_iter",300); params.setdefault("random_state",seed)
        est=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("classifier",MLPClassifier(**params))])
        est.fit(X_train,ytr); return TrainedClassifier(est,le.classes_,le,"mlp")
    raise ValueError(model_type)

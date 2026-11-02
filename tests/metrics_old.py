import numpy as np
from source_attribution.metrics import fpr95,oscr
from source_attribution.ood import energy_id_score,sme_id_score

def test_fpr95_perfect():
    y=np.r_[np.ones(100),np.zeros(100)];s=np.r_[np.linspace(.8,1,100),np.linspace(0,.2,100)]
    assert fpr95(y,s)==0.

def test_oscr_perfectish():
    y=np.array(["a","b","a","b"]);p=y.copy();ks=np.array([.9,.85,.88,.83]);us=np.array([.1,.2,.15])
    assert oscr(y,p,ks,us)>.95

def test_energy_orientation():
    assert energy_id_score(np.array([[8.,0.],[7.,0.]])).mean()>energy_id_score(np.array([[.1,.1],[0.,0.]])).mean()

def test_sme_finite():
    assert np.isfinite(sme_id_score(np.array([[2.,0.,-1.],[0.,0.,0.]]),4.)).all()

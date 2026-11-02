import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_fixed4_late_fusion_saved import choose_alpha

class TestFusionAlpha(unittest.TestCase):
    def test_validation_selects_best_alpha(self):
        y=np.array(['a','a','b','b'])
        classes=np.array(['a','b'])
        a=np.array([[.7,.3],[.8,.2],[.7,.3],[.6,.4]])
        b=np.array([[.9,.1],[.9,.1],[.1,.9],[.2,.8]])
        alpha, rows=choose_alpha(y,a,b,classes,[0,0.5,1])
        self.assertEqual(alpha,0.5)
        self.assertEqual(rows[1]['validation_macro_f1'],1.0)
    def test_tie_prefers_less_wavlm(self):
        y=np.array(['a','b'])
        classes=np.array(['a','b'])
        a=np.eye(2)*.95+.025
        b=a.copy()
        alpha,_=choose_alpha(y,a,b,classes,[0,.5,1])
        self.assertEqual(alpha,0)
if __name__=='__main__':unittest.main()

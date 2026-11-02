import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from compare_fixed4_aasist_with_baselines import score_pairs,holm_adjust

class Check(unittest.TestCase):
 def test_holm(self):
  self.assertEqual(holm_adjust([.02,.01,.5]),[.04,.03,.5])
 def test_paired_strata(self):
  n=40
  labels=['A','B']
  y=np.array(['A']*20+['B']*20)
  frame=pd.DataFrame({'truth':y,'aasist':y.copy(),
       'acoustic_lr':np.array(['B']*5+['A']*15+['B']*18+['A']*2),
       'wavlm_lr':np.array(['B']*7+['A']*13+['B']*17+['A']*3),
       'fusion':y.copy()})
  metrics,pairs,perclass=score_pairs(frame,labels,100,42)
  self.assertAlmostEqual(metrics['aasist']['macro_f1'],1)
  self.assertEqual(len(pairs),3)
  self.assertEqual(pairs[0]['only_first_correct'],7)
  self.assertEqual(pairs[0]['only_second_correct'],0)
  self.assertEqual(len(perclass),6)
 def test_zero_discordant(self):
  y=np.array(['A']*20+['B']*20)
  d=pd.DataFrame({'truth':y,'aasist':y,'fusion':y})
  _,pairs,_=score_pairs(d,['A','B'],100,2)
  self.assertEqual(pairs[0]['mcnemar_exact_p'],1)
  self.assertEqual(pairs[0]['delta_macro_f1'],0)

if __name__=='__main__':unittest.main()

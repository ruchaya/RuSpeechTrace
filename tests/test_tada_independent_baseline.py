"""Offline invariants; requires numpy, pandas and scikit-learn, not HF weights."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_tada_independent_baseline import (
    K,LAYER,MODEL, knn_predict, neighbors_cpu, calc_oscr,
    open_scores_metrics,subset_metrics,manifest_spec,read_cached_chunk,
    load_folds,find_mlaad_path,
)

class TestTadaOffline(unittest.TestCase):
    def test_original_code_settings(self):
        self.assertEqual((MODEL,LAYER,K),('facebook/w2v-bert-2.0',4,21))

    def test_sklearn_knn_euclidean_and_votes(self):
        from sklearn.neighbors import KNeighborsClassifier
        rng=np.random.default_rng(912)
        x=rng.normal(size=(110,12)).astype(np.float32)
        y=np.array(['B','A','C']*(len(x)//3)+['B','A'])
        q=rng.normal(size=(20,12)).astype(np.float32)
        m=KNeighborsClassifier(n_neighbors=K,metric='euclidean',weights='uniform',algorithm='brute')
        m.fit(x,y)
        dist,idx=neighbors_cpu(x,q,K,batch=7)
        pred,top3=knn_predict(y,idx)
        np.testing.assert_array_equal(pred,m.predict(q))
        self.assertEqual(top3.shape,(20,3))
        self.assertEqual(dist.shape,(20,K))

    def test_tie_break_lexicographic(self):
        labels=np.array(['z','a','z','a','b'])
        ind=np.array([[0,1,2,3]])
        pred,top3=knn_predict(labels,ind)
        self.assertEqual(pred[0],'a')
        self.assertEqual(top3[0].tolist(),['a','z','b'])

    def test_distances_separate_known_unknown(self):
        k=np.array([-0.1,-0.2,-0.15,-0.18])
        u=np.array([-6.0,-5.9,-5.8,-6.4])
        m=open_scores_metrics(np.array(['A']*4),np.array(['A']*4),k,u)
        self.assertAlmostEqual(m['auroc'],1)
        self.assertLess(m['fpr95'],1e-6)
        self.assertAlmostEqual(m['oscr'],1)
        self.assertAlmostEqual(m['aupr_ood'],1)

    def test_oscr_without_numpy_trapz(self):
        # Regression: recent NumPy builds no longer expose np.trapz.
        from unittest.mock import patch
        with patch.object(np, 'trapz', None, create=True):
            self.assertAlmostEqual(
                calc_oscr(
                    np.array(['A', 'A']),
                    np.array(['A', 'A']),
                    np.array([0.9, 0.8]),
                    np.array([0.2, 0.1]),
                ),
                1.0,
            )

    def test_subsets_and_paths(self):
        a=np.array(['a','b','b']);p=np.array(['a','b','a']);t=np.array([['a','b'],['b','a'],['b','a']])
        m=subset_metrics(a,p,t)
        self.assertAlmostEqual(m['accuracy'],2/3)
        self.assertAlmostEqual(m['top3'],1)
        x=find_mlaad_path(Path('D:/audio'),'./fake/ru/Model/sample.wav')
        self.assertEqual(x,Path('D:/audio/fake/ru/Model/sample.wav'))
        with self.assertRaises(ValueError): find_mlaad_path(Path('D:/audio'),'../../escape')

if __name__=='__main__':unittest.main()

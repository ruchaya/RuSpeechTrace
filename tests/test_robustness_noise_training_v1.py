import unittest
import numpy as np
import pandas as pd
from scripts.run_robustness_noise_training_v1 import select_train, CONDITIONS_TRAIN, part_path, CONDITIONS, paired_bootstrap
from scripts.run_robustness_main_v1 import add_noise
from pathlib import Path

class RobustnessLeakageUnitTests(unittest.TestCase):
    def test_stratified_frozen_selection(self):
        rows=[]
        for c in [f"class_{j:02d}" for j in range(34)]:
            for i in range(20):
                rows.append({"file_id":f"{c}_{i:03d}","generator":c,"split":"train","shard_path":"a.tar","audio_member":f"{i}.wav"})
            rows.append({"file_id":f"{c}_test","generator":c,"split":"test","shard_path":"a.tar","audio_member":"test.wav"})
        df=pd.DataFrame(rows)
        x=select_train(df,42,5); y=select_train(df,42,5)
        self.assertTrue(x.equals(y))
        self.assertEqual(len(x),170)
        self.assertTrue((x.generator.value_counts()==5).all())
        self.assertFalse(x.file_id.str.endswith("_test").any())
        self.assertEqual(len(set(x.file_id)),len(x))
    def test_noise_snr_and_determinism(self):
        rng=np.random.default_rng(111)
        audio=rng.normal(0,0.1,16000).astype('float32')
        for db, cond in ((30,'noise_30db'),(20,'noise_20db')):
            y=add_noise(audio,db,'a_train_file',cond)
            self.assertTrue(np.array_equal(y,add_noise(audio,db,'a_train_file',cond)))
            snr=20*np.log10(np.sqrt(np.mean(audio.astype(float)**2))/np.sqrt(np.mean((y-audio).astype(float)**2)))
            self.assertAlmostEqual(snr,db,places=3)
    def test_conditions_and_names(self):
        self.assertEqual(set(CONDITIONS_TRAIN), {'noise_30db','noise_20db'})
        self.assertNotEqual(part_path(Path('x'),'abc'),part_path(Path('x'),'abcd'))

    def test_paired_bootstrap_one_recording_not_two(self):
        # 34 balanced classes x 20 recordings, 6 conditions and 2 models.
        # The original script incorrectly selected 1360 (not 680) clean rows.
        items=[]
        classes=[f"class_{i:02d}" for i in range(34)]
        for cond in CONDITIONS:
            for class_idx, label in enumerate(classes):
                for j in range(20):
                    file_id=f"{label}_test_{j:02d}"
                    alternate=classes[(class_idx+1)%len(classes)]
                    for model in ("clean_trained", "augmented_trained"):
                        pred=(alternate if j < 4 and model=="clean_trained" else label)
                        items.append({"file_id":file_id,"generator":label,"condition":cond,
                                      "model":model,"predicted_generator":pred})
        predictions=pd.DataFrame(items).sample(frac=1,random_state=12).reset_index(drop=True)
        result=paired_bootstrap(predictions,["clean_trained","augmented_trained"],n_rep=8)
        self.assertEqual(len(result),len(CONDITIONS))
        for row in result:
            self.assertAlmostEqual(row["augmented_trained_macro_f1"],1.0,places=10)
            self.assertGreater(row["delta_macro_f1_aug_minus_clean"],0.0)
            self.assertEqual(len(row["delta_macro_f1_95ci"]),2)

if __name__=='__main__':unittest.main()

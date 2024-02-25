import unittest

import numpy as np
from ntanalysis.dvc_manager import dvc_load
from ntanalysis.infer import infer
from hydra import compose, initialize


class TestPredictions(unittest.TestCase):
    def testPredict(self):
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(
                config_name="config", overrides=["artifacts.enable_logger=false"]
            )
            dvc_load()
            predictions = infer(cfg)
            correct = np.sum(
                predictions["target_index"] == predictions["predicted_index"]
            )
            every = len(predictions)
            PRECISION = 0.7
            self.assertTrue(correct > PRECISION * every)

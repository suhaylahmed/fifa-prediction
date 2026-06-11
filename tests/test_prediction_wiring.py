import unittest

import numpy as np
import pandas as pd

from model.predict import _opposite_venue_mode, _predict_proba, _prediction_label


class _FixedBinaryModel:
    classes_ = np.array([0, 1])

    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def predict_proba(self, rows):
        positive = np.full(len(rows), self.positive_probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


class PredictionWiringTests(unittest.TestCase):
    def test_two_stage_probabilities_are_composed_into_three_classes(self):
        bundle = {
            "result_model_type": "two_stage",
            "draw_model": _FixedBinaryModel(0.20),
            "decisive_model": _FixedBinaryModel(0.75),
            "feature_columns": ["signal"],
        }

        probabilities = _predict_proba(bundle, pd.DataFrame({"signal": [1.0]}))

        np.testing.assert_allclose(probabilities, [0.20, 0.20, 0.60])

    def test_two_stage_label_uses_calibrated_draw_threshold(self):
        bundle = {"result_model_type": "two_stage", "draw_threshold": 0.37}

        label, policy = _prediction_label(bundle, np.array([0.30, 0.38, 0.32]))

        self.assertEqual(label, 1)
        self.assertIn("37%", policy)

    def test_mirrored_venue_preserves_home_team(self):
        self.assertEqual(_opposite_venue_mode("team_a_home"), "team_b_home")
        self.assertEqual(_opposite_venue_mode("team_b_home"), "team_a_home")
        self.assertEqual(_opposite_venue_mode("neutral"), "neutral")


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from dataclasses import replace

from display_defect_ai import Calibration, ImageInput, InferenceResult, Provenance


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.result = InferenceResult("example", 2, 1, 0.5, ((0.0, 1.0),),
                                      Provenance("test", "1", "test-v1", None, True))

    def test_uncalibrated_does_not_classify(self):
        self.assertEqual(self.result.decision, "uncalibrated")
        self.assertIsNone(self.result.to_dict()["calibration"])

    def test_threshold_boundary_and_both_sides(self):
        calibration = Calibration(0.5, "synthetic-unit-test", "boundary test", "synthetic-only")
        for score, expected in ((0.49, "normal"), (0.5, "anomaly"), (0.51, "anomaly")):
            with self.subTest(score=score):
                self.assertEqual(replace(self.result, score=score, calibration=calibration).decision, expected)

    def test_json_contract(self):
        payload = json.loads(json.dumps(self.result.to_dict(), allow_nan=False))
        self.assertEqual(set(payload), {"schema_version", "image_id", "width", "height", "score",
                                       "heatmap", "provenance", "calibration", "decision"})
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["heatmap"], [[0.0, 1.0]])
        self.assertTrue(payload["provenance"]["is_synthetic"])

    def test_absent_heatmap_is_supported(self):
        self.assertIsNone(replace(self.result, heatmap=None).to_dict()["heatmap"])

    def test_invalid_results(self):
        for fields in ({"score": float("nan")}, {"score": float("inf")}, {"score": True},
                       {"width": 0}, {"height": True}, {"image_id": " "},
                       {"heatmap": ((0.0,),)}, {"heatmap": ((-1.0, 0.0),)},
                       {"heatmap": ((float("nan"), 0.0),)}, {"heatmap": ((0.0, 1.1),)}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.result, **fields)

    def test_invalid_calibration(self):
        for threshold, method in ((float("inf"), "test"), (0.5, "")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                Calibration(threshold, "id", method, "validation")

    def test_invalid_input_and_provenance(self):
        with self.assertRaises(ValueError):
            ImageInput("image", 2, 1, ((0.0, 2.0),), True)
        with self.assertRaises(ValueError):
            ImageInput("image", 1, 1, ((0.0,),), "yes")
        with self.assertRaises(ValueError):
            Provenance("", "1", "preprocess", None, True)


if __name__ == "__main__":
    unittest.main()

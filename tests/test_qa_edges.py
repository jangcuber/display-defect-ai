"""Independent interface-v1 checks; synthetic arithmetic is not model evaluation."""

import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from display_defect_ai import Calibration, ImageInput, InferenceResult, Provenance
from display_defect_ai.synthetic import SyntheticSmokePredictor, read_synthetic_pgm


class QAContractEdges(unittest.TestCase):
    def setUp(self):
        self.image = ImageInput("qa-synthetic", 3, 2,
                                ((0.5, 0.75, 1.0), (0.0, 0.25, 0.5)), True)
        self.result = SyntheticSmokePredictor().predict(self.image)
        self.calibration = Calibration(0.5, "qa-only", "boundary exercise", "synthetic-validation")

    def test_nonfinite_and_nonnumeric_values_rejected_everywhere(self):
        for value in (math.nan, math.inf, -math.inf, True, False, "0.5", None):
            factories = {
                "score": lambda: replace(self.result, score=value),
                "threshold": lambda: replace(self.calibration, threshold=value),
                "pixel": lambda: ImageInput("qa", 1, 1, ((value,),), True),
                "heatmap": lambda: replace(self.result, width=1, height=1, heatmap=((value,),)),
            }
            for field, factory in factories.items():
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    factory()

    def test_shape_and_dimensions_rejected_for_inputs_and_results(self):
        for dimension in (0, -1, True, 2.0, "2"):
            for field in ("width", "height"):
                for obj in (self.image, self.result, replace(self.result, heatmap=None)):
                    with self.subTest(kind=type(obj).__name__, field=field, value=dimension):
                        with self.assertRaises(ValueError):
                            replace(obj, **{field: dimension})
        for grid in ((), ((), ()), ((0.5, 0.5),) * 3,
                     ((0.5, 0.5, 0.5), (0.5,)), ((0.5,) * 4,) * 2):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    replace(self.image, pixels=grid)
                with self.assertRaises(ValueError):
                    replace(self.result, heatmap=grid)

    def test_normalized_grid_range_is_distinct_from_score_range(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ImageInput("qa", 1, 1, ((value,),), True)
                with self.assertRaises(ValueError):
                    replace(self.result, width=1, height=1, heatmap=((value,),))
        for score in (-100.0, 100.0):
            with self.subTest(score=score):
                result = replace(self.result, score=score)
                self.assertEqual(result.to_dict()["score"], score)
                self.assertEqual(result.decision, "uncalibrated")

    def test_json_decision_matches_calibration_at_adjacent_float_boundary(self):
        for score, expected in ((math.nextafter(0.5, -math.inf), "normal"),
                                (math.nextafter(0.5, math.inf), "anomaly")):
            with self.subTest(score=score):
                result = replace(self.result, score=score, calibration=self.calibration)
                payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
                self.assertEqual(payload["decision"], expected)
                self.assertEqual(payload["decision"], result.decision)
                self.assertEqual(payload["calibration"], {
                    "threshold": 0.5, "calibration_id": "qa-only",
                    "method": "boundary exercise", "validation_split_id": "synthetic-validation",
                })
                self.assertEqual(replace(result, calibration=None).to_dict()["decision"], "uncalibrated")

    def test_rectangular_coordinates_identity_and_provenance_survive_json(self):
        payload = json.loads(json.dumps(self.result.to_dict(), allow_nan=False))
        self.assertEqual((payload["image_id"], payload["width"], payload["height"]),
                         (self.image.image_id, 3, 2))
        self.assertEqual(payload["heatmap"], [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(payload["provenance"], {
            "model_id": "synthetic-smoke-only", "model_version": "1",
            "preprocessing_id": "pgm-grayscale-divide-by-max-v1",
            "training_run_id": None, "is_synthetic": True,
        })
        self.assertIsNone(payload["calibration"])
        payload["heatmap"][0][0] = 1.0
        payload["provenance"]["model_id"] = "changed-copy"
        self.assertEqual(self.result.heatmap[0][0], 0.0)
        self.assertEqual(self.result.provenance.model_id, "synthetic-smoke-only")

    def test_required_provenance_and_calibration_text_cannot_be_blank(self):
        for obj, fields in ((self.result.provenance, ("model_id", "model_version", "preprocessing_id", "training_run_id")),
                            (self.calibration, ("calibration_id", "method", "validation_split_id"))):
            for field in fields:
                for value in ("", " \t\n"):
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        replace(obj, **{field: value})
        for obj in (self.image, self.result.provenance):
            for value in (0, 1, "true", None):
                with self.subTest(kind=type(obj).__name__, value=value), self.assertRaises(ValueError):
                    replace(obj, is_synthetic=value)
        real = Provenance("adapter-example", "1", "preprocess-example", "run-example", False)
        self.assertFalse(replace(self.result, provenance=real).to_dict()["provenance"]["is_synthetic"])

    def test_repeat_prediction_is_deterministic_and_does_not_modify_input(self):
        before = self.image.pixels
        predictor = SyntheticSmokePredictor()
        self.assertEqual(predictor.predict(self.image).to_dict(), predictor.predict(self.image).to_dict())
        self.assertEqual(self.image.pixels, before)

    def test_smoke_predictor_refuses_explicit_real_input(self):
        with self.assertRaises(ValueError):
            SyntheticSmokePredictor().predict(replace(self.image, is_synthetic=False))


    def test_equal_threshold_must_be_normal_under_agreed_strict_rule(self):
        # Canonical foundation 05b5748: anomaly iff score > threshold.
        provenance = Provenance("qa", "1", "qa", None, True)
        calibration = Calibration(0.5, "qa-only", "boundary exercise", "synthetic-validation")
        result = InferenceResult("qa", 1, 1, 0.5, None, provenance, calibration)
        payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(result.decision, "normal")
        self.assertEqual(payload["decision"], "normal")
        self.assertEqual(payload["score"], payload["calibration"]["threshold"])


class QASyntheticLoaderEdges(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa-synthetic.pgm"
            path.write_text(text, encoding="ascii")
            return read_synthetic_pgm(path)

    def test_rectangular_pgm_normalization_comments_and_public_basename(self):
        image = self.read("P2\n# SYNTHETIC_SMOKE_ONLY\n3 2 # width height\n4\n2 3 4\n0 1 2\n")
        self.assertEqual(image.image_id, "qa-synthetic.pgm")
        self.assertEqual((image.width, image.height), (3, 2))
        self.assertEqual(image.pixels, ((0.5, 0.75, 1.0), (0.0, 0.25, 0.5)))
        self.assertIs(image.is_synthetic, True)

    def test_pgm_limits_and_malformed_samples(self):
        for maximum in (1, 65535):
            with self.subTest(maximum=maximum):
                image = self.read(f"P2\n# SYNTHETIC_SMOKE_ONLY\n2 1\n{maximum}\n0 {maximum}\n")
                self.assertEqual(image.pixels, ((0.0, 1.0),))
        for body in ("1 1\n65536\n0", "1 1\n10\n-1", "1 1\n10\n0 1",
                     "1 1\n10\nnan", "1 1\n10\ninf", "1 1\n10\n0.5", "x 1\n10\n0"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.read("P2\n# SYNTHETIC_SMOKE_ONLY\n" + body)

    def test_missing_or_lookalike_marker_never_certifies_synthetic_origin(self):
        for marker in ("", "# synthetic_smoke_only\n", "# NOT_SYNTHETIC_SMOKE_ONLY\n",
                       "# SYNTHETIC_SMOKE_ONLY extra\n"):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                self.read("P2\n" + marker + "1 1\n10\n5\n")


class QAKnownContractDefects(unittest.TestCase):
    """Expected failures track foundation defects; remove decorators after core fixes.

    These are not passes. See docs/learning/qa-review.md for severity and repros.
    """

    @unittest.expectedFailure
    def test_missing_provenance_must_be_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            InferenceResult("qa", 1, 1, 0.5, None, None)

    @unittest.expectedFailure
    def test_mapping_calibration_must_be_rejected_at_construction(self):
        provenance = Provenance("qa", "1", "qa", None, True)
        with self.assertRaises(ValueError):
            InferenceResult("qa", 1, 1, 0.5, None, provenance, {"threshold": 0.5})

    @unittest.expectedFailure
    def test_non_string_image_id_must_raise_documented_value_error(self):
        with self.assertRaises(ValueError):
            ImageInput(None, 1, 1, ((0.5,),), True)

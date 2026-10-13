"""Headless checks: PYTHONPATH=src python -m unittest discover -s demo -v."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from display_defect_ai.contracts import Calibration
from display_defect_ai.synthetic import SyntheticSmokePredictor
from inspection import DemoSession, MAX_BYTES, load_image

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/synthetic"
VALID = b"P2\n# SYNTHETIC_SMOKE_ONLY\n2 1\n100\n50 100\n"


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private-name.pgm"
        self.path.write_bytes(VALID)
        self.session = DemoSession()

    def test_success_original_size_finite_score_and_private_id(self):
        result = self.session.inspect(self.path, synthetic=True)
        self.assertEqual((result.width, result.height), (2, 1))
        self.assertEqual(result.heatmap, ((0.0, 1.0),))
        self.assertEqual(self.session.image.pixels, ((0.5, 1.0),))
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.decision, "uncalibrated")
        self.assertIsNone(result.calibration)
        self.assertTrue(result.provenance.is_synthetic)
        self.assertIsNone(result.provenance.training_run_id)
        payload = json.dumps(result.to_dict(), allow_nan=False)
        self.assertNotIn("private-name", payload)
        self.assertNotIn(self.temp.name, payload)

    def test_bundled_examples_and_repeated_use(self):
        for _ in range(3):
            for name, score in (("bright_patch.pgm", 1.0), ("flat.pgm", 0.0)):
                result = self.session.inspect(FIXTURES / name, synthetic=True)
                self.assertEqual(result.score, score)
                self.assertEqual(result.decision, "uncalibrated")

    def test_rejected_inputs_clear_old_result_then_recover(self):
        invalid = [b"", b"P5\n# SYNTHETIC_SMOKE_ONLY\n1 1\n255\n0",
                   VALID.replace(b"# SYNTHETIC_SMOKE_ONLY\n", b""),
                   VALID.replace(b"2 1", b"257 1"), VALID.replace(b"2 1", b"0 1"),
                   VALID.replace(b"2 1", b"-1 1"), VALID.replace(b"2 1", b"1 257"),
                   VALID.replace(b"100\n50", b"0\n50"), VALID.replace(b"100\n50", b"65536\n50"),
                   VALID.replace(b"50 100", b"50"), VALID.replace(b"50 100", b"50 100 0"),
                   VALID.replace(b"50 100", b"-1 100"), VALID.replace(b"50 100", b"50 101"),
                   VALID.replace(b"50 100", b"nan 100"), VALID + b"\xff", b"x" * (MAX_BYTES + 1)]
        for content in invalid:
            with self.subTest(content=content[:70]):
                self.session.inspect(FIXTURES / "flat.pgm", synthetic=True)
                self.path.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.session.inspect(self.path, synthetic=True)
                self.assertIsNone(self.session.image)
                self.assertIsNone(self.session.result)
                self.path.write_bytes(VALID)
                self.session.inspect(self.path, synthetic=True)

    def test_file_types_missing_directory_and_os_error(self):
        for path in (self.path.with_suffix(".png"), self.path.parent / "absent.pgm", self.path.parent):
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_image(path)
        with patch.object(Path, "open", side_effect=PermissionError), self.assertRaises(OSError):
            self.session.inspect(self.path, synthetic=True)
        self.assertIsNone(self.session.result)

    def test_limits_inclusive_and_comments(self):
        self.path.write_bytes(b"P2\n# SYNTHETIC_SMOKE_ONLY\n256 256\n65535\n" + b"0 " * (256 * 256))
        self.assertEqual(load_image(self.path).width, 256)
        self.path.write_bytes(VALID + b"#" + b" " * (MAX_BYTES - len(VALID) - 1))
        self.assertEqual(load_image(self.path).width, 2)

    def test_real_mode_never_calls_predictor_and_clear_handles_cancel(self):
        self.session.inspect(self.path, synthetic=True)
        predictor = Mock()
        self.session.predictor = predictor
        with self.assertRaisesRegex(ValueError, "실제 추론 미지원"):
            self.session.inspect(self.path, synthetic=False)
        predictor.predict.assert_not_called()
        self.assertIsNone(self.session.result)
        self.session.clear()
        self.assertIsNone(self.session.image)

    def test_contract_mismatch_and_predictor_failure_clear_state(self):
        result = SyntheticSmokePredictor().predict(load_image(self.path))
        for invalid in (replace(result, image_id="wrong"),
                        replace(result, width=1, heatmap=((0.0,),)),
                        replace(result, provenance=replace(result.provenance, is_synthetic=False))):
            self.session.predictor = Mock(predict=Mock(return_value=invalid))
            with self.assertRaises(ValueError):
                self.session.inspect(self.path, synthetic=True)
            self.assertIsNone(self.session.result)
        self.session.predictor = Mock(predict=Mock(side_effect=RuntimeError("adapter error")))
        with self.assertRaises(RuntimeError):
            self.session.inspect(self.path, synthetic=True)
        self.assertIsNone(self.session.result)

    def test_optional_heatmap_and_calibration_are_preserved(self):
        result = SyntheticSmokePredictor().predict(load_image(self.path))
        for calibration, decision in ((None, "uncalibrated"),
                                      (Calibration(1.0, "unit-test-only", "unit-test", "unit-test"), "anomaly")):
            expected = replace(result, heatmap=None, calibration=calibration)
            self.session.predictor = Mock(predict=Mock(return_value=expected))
            actual = self.session.inspect(self.path, synthetic=True)
            self.assertIs(actual, expected)
            self.assertIsNone(actual.heatmap)
            self.assertEqual(actual.decision, decision)


if __name__ == "__main__":
    unittest.main()

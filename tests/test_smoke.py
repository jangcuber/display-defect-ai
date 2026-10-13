import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

from display_defect_ai.synthetic import SyntheticSmokePredictor, read_synthetic_pgm

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/synthetic"


class SyntheticSmokeTests(unittest.TestCase):
    """Only wiring, validation, and deterministic synthetic arithmetic are tested."""

    def test_flat_and_patch(self):
        predictor = SyntheticSmokePredictor()
        flat = predictor.predict(read_synthetic_pgm(FIXTURES / "flat.pgm"))
        patch = predictor.predict(read_synthetic_pgm(FIXTURES / "bright_patch.pgm"))
        self.assertEqual(flat.score, 0.0)
        self.assertEqual(patch.score, 1.0)
        self.assertEqual(patch.heatmap[3][3], 1.0)
        self.assertEqual(patch.heatmap[0][0], 0.0)
        self.assertEqual(patch.decision, "uncalibrated")
        self.assertIsNone(patch.provenance.training_run_id)
        self.assertTrue(patch.provenance.is_synthetic)

    def test_non_synthetic_rejected(self):
        image = read_synthetic_pgm(FIXTURES / "flat.pgm")
        with self.assertRaises(ValueError):
            SyntheticSmokePredictor().predict(replace(image, is_synthetic=False))

    def test_invalid_pgm(self):
        for content in ("P2\n1 1\n100\n50", "P5\n# SYNTHETIC_SMOKE_ONLY\n1 1\n100\n50",
                        "P2\n# SYNTHETIC_SMOKE_ONLY\n0 1\n100\n50",
                        "P2\n# SYNTHETIC_SMOKE_ONLY\n1 1\n0\n0",
                        "P2\n# SYNTHETIC_SMOKE_ONLY\n2 1\n100\n50",
                        "P2\n# SYNTHETIC_SMOKE_ONLY\n1 1\n100\n101"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.pgm"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_synthetic_pgm(path)

    def test_cli_json_and_error(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        command = [sys.executable, "-m", "display_defect_ai.cli"]
        success = subprocess.run(command + [str(FIXTURES / "bright_patch.pgm")],
                                 env=env, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        result = json.loads(success.stdout)
        self.assertEqual(result["decision"], "uncalibrated")
        self.assertTrue(result["provenance"]["is_synthetic"])
        failure = subprocess.run(command + [str(FIXTURES / "missing.pgm")],
                                 env=env, capture_output=True, text=True)
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(failure.stdout, "")
        self.assertNotIn("Traceback", failure.stderr)


if __name__ == "__main__":
    unittest.main()

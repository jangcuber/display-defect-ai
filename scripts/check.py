"""Run core, independent QA, and headless demo tests without hidden skips."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    suite = unittest.TestSuite()
    for directory in ("tests", "demo"):
        suite.addTests(unittest.TestLoader().discover(str(ROOT / directory)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    complete = result.wasSuccessful() and not result.skipped and not result.expectedFailures
    print(f"Aggregate: {result.testsRun} tests; failures={len(result.failures)}; "
          f"errors={len(result.errors)}; skipped={len(result.skipped)}; "
          f"expected_failures={len(result.expectedFailures)}; "
          f"unexpected_successes={len(result.unexpectedSuccesses)}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

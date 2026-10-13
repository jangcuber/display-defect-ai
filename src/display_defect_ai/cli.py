"""Emit a JSON smoke result without model downloads or training."""

import argparse
import json

from .synthetic import SyntheticSmokePredictor, read_synthetic_pgm


def main() -> None:
    parser = argparse.ArgumentParser(description="SYNTHETIC smoke test only; no real defect validation")
    parser.add_argument("image", help="marked synthetic ASCII .pgm fixture")
    args = parser.parse_args()
    try:
        result = SyntheticSmokePredictor().predict(read_synthetic_pgm(args.image))
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

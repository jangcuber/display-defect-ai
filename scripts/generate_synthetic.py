"""Recreate committed synthetic fixtures deterministically; no external data."""

from pathlib import Path

DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/synthetic"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, changed in (("flat", False), ("bright_patch", True)):
        rows = [[50] * 8 for _ in range(8)]
        if changed:
            for y in (3, 4):
                for x in (3, 4):
                    rows[y][x] = 100
        content = "P2\n# SYNTHETIC_SMOKE_ONLY\n8 8\n100\n"
        content += "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
        (DESTINATION / f"{name}.pgm").write_text(content, encoding="ascii")


if __name__ == "__main__":
    main()

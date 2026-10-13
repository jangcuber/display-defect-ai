"""Synthetic plumbing only: this is NOT PatchCore or a trained detector."""

from pathlib import Path

from .contracts import ImageInput, InferenceResult, Provenance


def read_synthetic_pgm(path: str | Path) -> ImageInput:
    """Read our tiny ASCII PGM fixtures, requiring an explicit synthetic marker."""
    path = Path(path)
    text = path.read_text(encoding="ascii")
    if "# SYNTHETIC_SMOKE_ONLY" not in text.splitlines():
        raise ValueError("fixture requires SYNTHETIC_SMOKE_ONLY marker")
    tokens = " ".join(line.split("#", 1)[0] for line in text.splitlines()).split()
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError("only ASCII P2 PGM fixtures are supported")
    width, height, maximum = map(int, tokens[1:4])
    if width <= 0 or height <= 0 or not 1 <= maximum <= 65535:
        raise ValueError("invalid PGM dimensions or maximum")
    samples = [int(token) for token in tokens[4:]]
    if len(samples) != width * height or any(not 0 <= v <= maximum for v in samples):
        raise ValueError("invalid PGM sample count or value")
    pixels = tuple(tuple(v / maximum for v in samples[y * width:(y + 1) * width]) for y in range(height))
    return ImageInput(path.name, width, height, pixels, is_synthetic=True)


class SyntheticSmokePredictor:
    """Measure deviation from a fixed gray field, solely to exercise the interface."""

    def predict(self, image: ImageInput) -> InferenceResult:
        if not image.is_synthetic:
            raise ValueError("synthetic smoke predictor rejects non-synthetic inputs")
        heatmap = tuple(tuple(abs(value - 0.5) * 2 for value in row) for row in image.pixels)
        return InferenceResult(
            image_id=image.image_id, width=image.width, height=image.height,
            score=max(max(row) for row in heatmap), heatmap=heatmap,
            provenance=Provenance(
                model_id="synthetic-smoke-only", model_version="1",
                preprocessing_id="pgm-grayscale-divide-by-max-v1",
                training_run_id=None, is_synthetic=True,
            ),
        )

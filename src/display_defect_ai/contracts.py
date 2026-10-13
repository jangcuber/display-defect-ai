"""Version 1 contract. Scores are anomaly strength, never probabilities."""

from dataclasses import asdict, dataclass
import math
from typing import Protocol

Grid = tuple[tuple[float, ...], ...]


def _finite(value: float) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _grid(values: Grid, width: int, height: int) -> None:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if not isinstance(values, (tuple, list)) or any(not isinstance(row, (tuple, list)) for row in values):
        raise ValueError("grid must be a sequence of rows")
    if len(values) != height or any(len(row) != width for row in values):
        raise ValueError("grid must have height rows and width columns")
    if any(not _finite(v) or not 0 <= v <= 1 for row in values for v in row):
        raise ValueError("grid values must be finite numbers in [0, 1]")


@dataclass(frozen=True)
class ImageInput:
    """Grayscale [0,1], top-left origin, pixels[y][x]; RGB adapter is future work."""

    image_id: str
    width: int
    height: int
    pixels: Grid
    is_synthetic: bool

    def __post_init__(self) -> None:
        if not _text(self.image_id) or type(self.is_synthetic) is not bool:
            raise ValueError("image_id and explicit boolean is_synthetic are required")
        _grid(self.pixels, self.width, self.height)


@dataclass(frozen=True)
class Calibration:
    threshold: float
    calibration_id: str
    method: str
    validation_split_id: str

    def __post_init__(self) -> None:
        if not _finite(self.threshold):
            raise ValueError("threshold must be finite")
        if not all(_text(v) for v in (self.calibration_id, self.method, self.validation_split_id)):
            raise ValueError("calibration requires method and validation provenance")


@dataclass(frozen=True)
class Provenance:
    model_id: str
    model_version: str
    preprocessing_id: str
    training_run_id: str | None
    is_synthetic: bool

    def __post_init__(self) -> None:
        if not all(_text(v) for v in (self.model_id, self.model_version, self.preprocessing_id)):
            raise ValueError("model and preprocessing identifiers are required")
        if type(self.is_synthetic) is not bool:
            raise ValueError("is_synthetic must be a boolean")
        if self.training_run_id is not None and not _text(self.training_run_id):
            raise ValueError("training_run_id must be nonempty or None")


@dataclass(frozen=True)
class InferenceResult:
    image_id: str
    width: int
    height: int
    score: float
    heatmap: Grid | None
    provenance: Provenance
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        if not _text(self.image_id) or not _finite(self.score):
            raise ValueError("image_id and finite score are required")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance object")
        if self.calibration is not None and not isinstance(self.calibration, Calibration):
            raise ValueError("calibration must be a Calibration object or None")
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers")
        if self.heatmap is not None:
            _grid(self.heatmap, self.width, self.height)

    @property
    def decision(self) -> str:
        if self.calibration is None:
            return "uncalibrated"
        return "anomaly" if self.score > self.calibration.threshold else "normal"

    def to_dict(self) -> dict:
        """JSON-ready payload; null calibration means no threshold decision."""
        result = asdict(self)
        result.update(schema_version="1.0", decision=self.decision)
        if self.heatmap is not None:
            result["heatmap"] = [list(row) for row in self.heatmap]
        return result


class Predictor(Protocol):
    def predict(self, image: ImageInput) -> InferenceResult:
        """Return one image result; higher score means more anomalous."""
        ...

"""Small, dependency-free inference contract for a student project."""

from .contracts import Calibration, ImageInput, InferenceResult, Predictor, Provenance

__all__ = ["Calibration", "ImageInput", "InferenceResult", "Predictor", "Provenance"]

"""Bounded local file input and session state; no GUI or third-party imports."""

from pathlib import Path
import os
import stat

from display_defect_ai.contracts import ImageInput, InferenceResult, Predictor
from display_defect_ai.synthetic import SyntheticSmokePredictor

MAX_BYTES = 1024 * 1024
MAX_SIDE = 256
UNSUPPORTED = "실제 추론 미지원: 실제 데이터와 학습된 모델이 없습니다. 합성 예제로만 실행하세요."


def load_image(path: str | Path) -> ImageInput:
    """Accept only small, explicitly marked synthetic P2 PGM files."""
    path = Path(path)
    if path.suffix.lower() != ".pgm":
        raise ValueError(".pgm 합성 파일만 지원합니다. PNG/JPEG 및 실제 사진은 지원하지 않습니다.")
    if not path.is_file():
        raise ValueError("일반 이미지 파일을 선택하세요.")
    with path.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("일반 이미지 파일을 선택하세요.")
        raw = stream.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("파일은 1바이트 이상, 1 MiB 이하여야 합니다.")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("ASCII P2 PGM만 지원합니다.") from exc
    if "# SYNTHETIC_SMOKE_ONLY" not in lines:
        raise ValueError("합성 표시(# SYNTHETIC_SMOKE_ONLY)가 필요합니다. 실제 사진에 표시를 붙이지 마세요.")
    tokens = " ".join(line.split("#", 1)[0] for line in lines).split()
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError("ASCII P2 PGM 헤더가 필요합니다.")
    try:
        width, height, maximum = map(int, tokens[1:4])
        if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
            raise ValueError
        if not 1 <= maximum <= 65535 or len(tokens) - 4 != width * height:
            raise ValueError
        samples = [int(value) for value in tokens[4:]]
        if any(value < 0 or value > maximum for value in samples):
            raise ValueError
    except ValueError as exc:
        raise ValueError("PGM 값 오류: 가로·세로 1–256, 최댓값 1–65535, 픽셀 수와 범위를 확인하세요.") from exc
    pixels = tuple(tuple(value / maximum for value in samples[y * width:(y + 1) * width])
                   for y in range(height))
    # Do not propagate a private filename or full path into result provenance.
    return ImageInput("synthetic-upload", width, height, pixels, is_synthetic=True)


class DemoSession:
    """Clear stale output before each selection, cancellation, or failed attempt."""

    def __init__(self, predictor: Predictor | None = None):
        self.predictor = predictor if predictor is not None else SyntheticSmokePredictor()
        self.clear()

    def clear(self) -> None:
        self.image: ImageInput | None = None
        self.result: InferenceResult | None = None

    def inspect(self, path: str | Path, *, synthetic: bool) -> InferenceResult:
        self.clear()
        if not synthetic:
            raise ValueError(UNSUPPORTED)
        image = load_image(path)
        result = self.predictor.predict(image)
        if (result.image_id, result.width, result.height) != (image.image_id, image.width, image.height):
            raise ValueError("모델 출력의 식별자 또는 원본 크기가 입력과 다릅니다.")
        if result.provenance.is_synthetic != image.is_synthetic:
            raise ValueError("모델 출력의 합성 출처가 입력과 다릅니다.")
        self.image, self.result = image, result
        return result

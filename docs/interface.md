# 제안 인터페이스 v1.0

공유 기준은 `src/display_defect_ai/contracts.py`입니다. 외부 라이브러리 없이 import할 수 있습니다.
데모는 `Predictor.predict(image: ImageInput) -> InferenceResult`만 호출합니다.
모델 담당은 이 프로토콜을 만족하는 어댑터를 제공합니다. 기존 Tkinter 합성 GUI는 이 계약을 사용합니다.

## 입력

`ImageInput(image_id, width, height, pixels, is_synthetic)`.
`pixels[y][x]`는 [0,1] 범위의 회색조 값입니다. 원점은 왼쪽 위이고 배열은 height행 × width열입니다.
`image_id`는 공개 가능한 식별자만 쓰고 로컬 전체 경로나 개인 식별자는 넣지 않습니다.
`is_synthetic`은 호출자가 명시합니다. 초기 파일 로더는 합성 표시가 있는 ASCII PGM만 지원합니다.
실제 PNG/JPEG/RGB 디코딩, 크기 제한과 채널 전처리는 데이터 결정 후 별도 어댑터로 추가합니다.
RGB가 필요하면 버전 변경을 합의하고 이 회색조 계약을 조용히 바꾸지 않습니다.

## 출력 JSON

| 필드 | 의미 |
|---|---|
| `schema_version` | 현재 `"1.0"` |
| `image_id`, `width`, `height` | 원본 입력 식별자와 크기 |
| `score` | 유한한 수, 클수록 이상함. 범위는 모델마다 다르며 확률이 아님 |
| `decision` | `uncalibrated`, `normal`, `anomaly` 중 하나 |
| `calibration` | 미보정 시 null. 보정 시 threshold, calibration_id, method, validation_split_id |
| `heatmap` | null 또는 [0,1]의 height × width 2차원 배열, 클수록 이상 위치 |
| `provenance` | model_id, model_version, preprocessing_id, training_run_id, is_synthetic |

`calibration is None`이면 언제나 `uncalibrated`입니다. 보정 정보가 있으면 **`score > threshold`**일 때
`anomaly`, 임계값과 같거나 작으면 `normal`입니다. 판정을 따로 저장하거나 데모가 임계값을 임의로 정하지 않습니다.
초기 foundation의 `>=` 규칙을 통합 전 검토에서 엄격한 `>`로 수정했습니다. 초기 커밋을 사용한
소비자는 동점 판정과 경계 테스트를 갱신해야 합니다. 필드와 schema_version `1.0`은 그대로 유지합니다.
보정 정보에는 검증 분할과 방법을 기록합니다. 최종 테스트를 보고 임계값을 바꾸지 않습니다.
단위 테스트의 임계값은 분기 확인용이며 실제 데이터 보정 결과가 아닙니다.

히트맵은 원본 좌표로 되돌린 표시용 값입니다. 어댑터는 리사이즈/보간/정규화 방식을
`preprocessing_id`가 가리키는 설정에 고정하고 기록합니다. 이미지마다 히트맵 최대값을 1로
만드는 방식은 서로 비교할 수 없으므로 사용할 경우 화면에 명시해야 합니다. 히트맵 밝기를
결함 확률로 설명하거나 히트맵의 최대값을 모든 모델의 `score`라고 가정하지 않습니다.
위치 출력을 제공하지 못하면 null을 반환하고 화면은 “위치 정보 없음”을 표시합니다.

`training_run_id=None`은 학습되지 않았음을 뜻합니다. 실제 모델에서는 가중치 해시, 라이브러리 버전,
정상 학습 분할, 시드와 설정을 담은 로컬 실행 기록을 참조합니다. `is_synthetic`은 입력 출처이며
모델의 종류는 `model_id`로 구분합니다. 현재 대역은 `synthetic-smoke-only`이고 실제 입력을 거부합니다.

## 데모 세션용 예제

```python
from display_defect_ai.synthetic import SyntheticSmokePredictor, read_synthetic_pgm

predictor = SyntheticSmokePredictor()
image = read_synthetic_pgm("tests/fixtures/synthetic/bright_patch.pgm")
payload = predictor.predict(image).to_dict()
assert payload["decision"] == "uncalibrated"
assert payload["provenance"]["is_synthetic"] is True
```

화면에는 항상 “합성 테스트 / 실제 성능 미검증” 표시를 유지하세요. 잘못된 입력은 `ValueError`,
파일 접근 오류는 `OSError`로 전달합니다. 데모는 오류를 사용자 안내로 변환합니다.
생성자는 필수 문자열·배열 구조·수치 범위를 검증합니다. `provenance`는 `Provenance` 객체,
`calibration`은 `Calibration` 객체 또는 None이어야 합니다. JSON dict를 생성자에 그대로 전달하지 말고
필요한 객체를 명시적으로 구성하세요. 잘못된 타입도 생성 시 `ValueError`로 거부합니다.
필드 이름/좌표/점수 의미가 바뀌면 리더와 합의하고 계약·테스트·소비자를 함께 수정합니다.

## 학습 데이터 계약 v1 (기존 추론 계약과 별개)

- `images/{train,val,test}/*.png`: RGB 의미의 컬러 이미지(OpenCV 배열은 BGR), 현재 정사각 512픽셀.
- `labels/{train,val,test}/*.txt`: 같은 stem, 행마다 `class cx cy width height`.
  좌표는 최종 촬영 이미지 크기로 나눈 [0,1] 값. 클래스 0=dot, 1=line.
  정상은 별도 클래스가 아니라 빈 라벨 파일입니다.
- `manifest.jsonl`: 이미지별 source_case_id, content_id, split, 생성 조건, 원근 행렬,
  원본 결함 조건, 최종 픽셀 bbox_xyxy를 기록합니다. x2/y2는 포함하지 않는 상한입니다.
- 박스는 원근 변환한 결함 마스크에 흐림 여유 2픽셀을 더한 외접 사각형입니다.
- `dataset.yaml`: YOLO 데이터 경로/클래스. 이동 후 path를 새 데이터 루트로 바꿉니다.

실제 YOLO 추론은 별도 detection 계약으로 boxes/class/confidence와 원본 좌표를 다뤄야 합니다.
현재 회색조 v1의 score/heatmap에 검출 결과를 억지로 넣지 않습니다. detection 출력 구현은 다음 세션에서
공유 계약·소비자와 함께 추가합니다. confidence는 불량 확률로 해석하지 않습니다.
사진별 종합 판정은 검증한 클래스별 confidence/크기/개수 등의 규칙과 rule_id를 기록해야 하며 아직 미구현입니다.

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
사진별 종합 판정은 검증한 클래스별 confidence/크기/개수 등의 규칙과 rule_id를 기록해야 합니다.
기존 맥 앱의 detection 계약은 아래 OLED v2를 따릅니다. 현재 앱 v0.4는 뒤의 OLED v3를 사용합니다.

## OLED 흰 화면 detection 계약 v2

맥 검사 앱과 `scripts/inspect_white_photo.py`는 기존 회색조 v1과 별개인
`schema_version="oled-white-2"`를 사용합니다.

- 클래스 ID는 0=`dark_dot`, 1=`color_dot`, 2=`horizontal_line`, 3=`vertical_line`, 4=`dark_region`입니다.
- 정상 사진의 라벨 파일은 비어 있습니다. 불량 사진에는 여러 종류와 여러 객체가 포함될 수 있습니다.
- 새 manifest의 `objects`는 객체별 `kind`, `bbox_xyxy`, 생성 시 `attributes`를 기록합니다.
  군집점은 점별로, 끊긴 선은 하나의 객체로 표시합니다. 기존 `bbox_xyxy` 배열과 순서가 같습니다.
- 사진별 결과의 `decision`은 `normal`, `defect`, `unavailable`입니다. `rule_id`는
  `oled-white-five-either-detector-v1`이며 두 검출기 중 하나라도 객체가 있으면 `defect`입니다.
- `opencv`와 `yolo`는 객체 배열입니다. 각 객체는 `kind`, `title`, `score`, `polygon`, `detail`,
  `attributes`를 가집니다. `polygon`은 EXIF 방향을 적용한 원본 사진 크기로 정규화한 네 꼭짓점입니다.
- `score`는 YOLO에서는 모델 confidence, OpenCV에서는 8비트 밝기 차이입니다. 두 점수는 서로 비교하지 않습니다.
- 앱의 `attributes`는 합성 manifest를 읽지 않고 검출된 사진에서 계산합니다. 관찰 색, 군집,
  선 형태, 512픽셀 정렬 영상의 박스 폭·높이를 담으며 전기적 원인을 진단하지 않습니다.
- 모델과 임계값은 검증 분할에서 고릅니다. `validation_selection.json`에 클래스별 confidence,
  선 중복 억제 IoU, OpenCV 설정과 가중치 SHA-256을 기록하고 앱은 해시가 일치하는 모델을 사용합니다.
- 입력 실패·정렬 실패·추론 실패는 `unavailable`로 처리하고 앱에 검사 불가를 표시합니다.

### 얇은 선 다중 해상도 추론

`validation_selection.json`의 `selected.size`는 기본 추론 크기이며 선택 필드 `line_sizes`는
선 후보를 얻을 추가 입력 크기 배열입니다. 기존 v2 앱은 `size=640`, `line_sizes=[960]`을 사용했습니다.
배열이 없으면 기존 단일 해상도 경로로 동작합니다. 추가 추론의 점·영역 후보는 사용하지 않습니다.
같은 종류의 선 후보 중 축 위치·두께·길이 겹침 조건을 만족하는 박스를 길이 방향으로 결합한 뒤
클래스별 confidence와 선 중복 억제를 적용합니다. OpenCV 후보나 정답 라벨로 보정하지 않습니다.

앱과 시험 수집은 `oled_yolo.predict_candidates`를 공유합니다. 추론 캐시는 가중치·manifest 해시와
`image_size`, `line_sizes`, `minimum_confidence`를 기록합니다. 다른 해상도·파이프라인의 캐시는
검증 설정과 함께 사용할 수 없습니다. 기존 캐시의 생략 필드는 640, 빈 배열, 0.01로 해석합니다.
현재 후보 수집 하한은 0.001이고 최종 임계값은 검증 자료에서 고릅니다. 출력 v2의 필드는 유지합니다.

### 짝 데이터 v2의 타일·전체 장면 좌표

`data/oled-white-paired-v2/`의 5종 클래스 ID는 동일합니다. `kind=mixed`는 사진의 생성 그룹이며
추가 모델 클래스가 아닙니다. 복합 사진의 `objects`와 YOLO 라벨에는 실제 각 객체의 기존 클래스가 들어갑니다.

- `images/{train,val,test}`의 PNG는 1024×1024의 실제 학습 타일입니다. `objects[].bbox_xyxy`는
  이 타일의 픽셀 좌표, YOLO 라벨은 타일 크기로 정규화한 좌표입니다.
- `normal_image`는 같은 source_case_id·split·tile_xyxy·좌표별 노이즈를 공유하는 정상 PNG입니다.
- `tile_xyxy`는 고정 구도 전체 장면 5888×3368 안에서의 타일 범위입니다. 전체 장면의
  좌표에 접근할 때 타일 원점을 더합니다. 타일을 원본 전체 사진으로 해석하지 않습니다.
- `scene_objects`는 전체 장면을 재현하는 결함 조건입니다. 그 안의 x/y/폭/길이는 합성 패널
  1920×1080 좌표이며, 카메라 대응은 축별 3배·테두리는 64픽셀입니다. 실제 장비 보정값은 아닙니다.
- 점은 1~3픽셀 크기이고 군집점은 각각 라벨링합니다. 선 폭은 1~2픽셀이며 끊긴 선은 한 객체입니다.
  채널별 강도·공간 분포는 `channel_strengths`, `channel_profiles`로 기록합니다.
- 객체 박스는 같은 정상과 단독 결함 영상의 최대 RGB 차이가 2 DN 이상인 픽셀의 외접 박스입니다.
  물리 마스크나 앱 테두리의 크기와 다릅니다. 긴 선의 `clipped_by_tile`은 전체 선이 타일 밖까지
  이어진다는 뜻이며, 저장 라벨은 타일 안에서 보이는 부분을 가리킵니다.
- 모든 객체의 native_bbox를 촬영 좌표로 옮기고 흐림 범위를 포함한 영역 밖에서는 정상과 불량의
  픽셀이 같습니다. 복합 객체의 겹친 출력 저하는 투과율 곱으로 합성합니다.
- `source_cases.jsonl`과 버전 고정된 렌더러 조건으로 전체 장면을 재현합니다. 좌표별 난수와
  고정 합산 순서의 흐림을 사용해 전체 내보내기의 같은 위치가 저장 타일과 정확히 일치합니다.
- 데이터 전체·파일·픽셀 해시와 라이브러리 버전을 기록합니다. 원본별 분할이므로 파생 사진이
  학습/검증/시험을 넘나들지 않습니다. 이전 512픽셀 앱 전처리와 시험 설정을 그대로 적용해
  성능을 주장하지 않으며, 이 변경은 기존 추론 JSON의 필드를 바꾸지 않습니다.

### 원본 해상도 판독 v3

새 경로 `scripts/inspect_native_photo.py`는 `oled-white-3` 결과를 사용합니다. 기존 v1/v2와
표준 라이브러리 계약 테스트용 구현은 유지합니다. 검증에서 선택해 동결한 U-Net 또는 YOLO26n을
한 번 로드하고, 정상 짝·합성 조건·정답 마스크 없이 입력 RGB 사진만 추론합니다.

- 설정은 `outputs/paired-model-comparison-v1/frozen_selection.json`입니다. 가중치와 추론 코드 해시가
  다르면 실행을 거부합니다. 앱과 평가가 `native_inference.py`의 같은 후처리를 사용합니다.
- 1024픽셀 타일, 128픽셀 중첩을 사용하며 입력 사진을 축소하거나 색 보정하지 않습니다.
  전체 합성 구도 5888×3368에서는 테두리 64픽셀과 광학 흐림 전이 3픽셀을 제외한 고정 ROI를 검사합니다.
- 결과의 `width/height`, `detections[].box`, `crop_box`는 원본 이미지 픽셀 좌표입니다.
  `polygon`만 사진 크기로 정규화합니다. `detail`의 검출 박스 크기는 물리적 결함 크기가 아닙니다.
- `decision`은 검증 조건 안에서 검출이 있으면 `defect`, 없으면 `normal`입니다. 크기·구도·노출이
  검증 조건을 벗어나면 `review`이며 정상/불량을 확정하지 않습니다. 입력·추론 오류는 오류 이벤트입니다.
- `model_name`, `selected_model`, `device`, `seconds`, `profile_ok`가 모델과 실행 조건을 기록합니다. `weights_sha256`과 `inference_source_sha256`은 실제 로드한 가중치·추론 코드의 식별값입니다.
- `preview`는 표시용 축소 사진입니다. `detections[].crop`은 표시 없는 원본 픽셀 부분 사진이며,
  점·선·영역의 원본 부분 사진은 각각 최대 64·128·256픽셀입니다. 긴 선/큰 영역은 일부분을 저장하며, U-Net에서는 중간이 빈 곳이면 가장 가까운 양성 출력 픽셀을 중심으로 삼습니다. `crop_anchor`도 원본 좌표입니다. 앱은 1·2·4·8배 표시를 지원합니다.
- U-Net의 `pixel_mask.png`는 클래스별 임계값을 넘은 출력의 독립 비트 PNG입니다. YOLO 박스로
  픽셀 마스크를 꾸미지 않습니다. 선택적으로 표시하는 `mask_preview`는 모델의 픽셀 응답이며,
  컴포넌트 크기·모델 점수로 걸러진 최종 검출 목록과 구분합니다.
- `--server`에서는 요청 `{id, photo, output}`을 한 줄 JSON으로 받습니다. stdout은
  `loading`, `progress(done,total)`, `result(result)`, `error(message)` 이벤트만 전송합니다.
  같은 프로세스가 여러 사진을 처리해 모델 재로딩을 줄이며, 취소 시 해당 작업 프로세스를 종료합니다.

정답 마스크는 `masks/{split}/{image_id}.png`의 uint8 비트 플레인입니다. 클래스 i는 `1<<i`이며
겹치는 결함의 여러 비트를 보존합니다. 같은 정상에 단독 결함을 넣었을 때 최대 RGB 차이가
2 DN 이상인 픽셀을 포함합니다. `mask_manifest.jsonl`에 파일 해시와 클래스별 픽셀 수를,
`segmentation.json`에 생성 기준과 데이터·마스크 목록 해시를 기록합니다.

전체 구도 판정에서는 각 변의 두 위치에서 테두리 전이를 확인합니다. 여덟 지점의 경계가 예상 위치 64픽셀에 일치하고 바깥이 어둡고 안쪽이 밝아야 등록된 구도로 간주합니다. 모델 입력을 정렬·보간해 맞추지 않으며, 구도가 다르면 확인 필요로 표시합니다.

영역의 타일 경계 박스 처리는 `merge_regions`와 `region_iou`로 동결합니다. 두 모델 모두 같은 전체 검증 사진에서 중복 억제 또는 겹친 박스 합치기와 IoU 기준 0.35·0.25·0.15 중 선택합니다. 앱과 최종 시험은 선택된 동일 후처리를 적용합니다.

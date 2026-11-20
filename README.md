# Display Defect AI — 학습 프로젝트

한 달 안에 혼자 데이터 생성 → 모델 학습 → 맥 검사 → 평가를 한 번 완성한 뒤 팀원용 템플릿을 만듭니다.
현재 대상은 OLED 생산 직후 흰 화면의 어두운 점·색 이상점·가로선·세로선·밝기·색 영역 저하 5종입니다.
OpenCV 규칙 기준을 만든 뒤, 작은 점·얇은 선·완만한 영역에 대해 U-Net 픽셀 분할과 YOLO26n 객체 검출을 비교했습니다.
사진별 판정은 검증 분할에서 정한 규칙으로 종합합니다. 합성 영역 저하는 실제 무라의 보정된 물리 모델이 아닙니다.

학습용 정상·점·선 PNG와 YOLO 라벨 생성기는 [데이터 생성 인계](docs/generation-handoff.md)를 참고하세요.
기존 8×8 PGM 데모는 계약 테스트 대역으로 유지하며 새 학습 데이터와 구분합니다.

## 맥에서 흰 화면 검사하기

**[GitHub Releases에서 앱 다운로드](https://github.com/jangcuber/display-defect-ai/releases/tag/v0.4.0)** · Apple Silicon 맥용입니다. ZIP을 풀고 앱에서 **예시 사진 → 복합 불량**을 선택하면 검사 흐름을 볼 수 있습니다.
번들 설치 조건과 개발용 소스 빌드 방법은 [맥 앱 배포 안내](docs/mac-release.md)에 있습니다.

현재 앱(v0.4)은 검증 성능으로 선택한 **U-Net**으로 원본 해상도의 흰 화면 PNG/JPEG를 검사합니다.
표시 없는 원본을 기본으로 보여 주며 검출 위치와 실제 모델의 픽셀 출력을 선택해 볼 수 있습니다.
목록에서 결함을 누르면 원본 부분을 1·2·4·8배로 확대하고 종류·개수·박스 크기·모델 점수를 표시합니다.
모델 점수는 불량 확률이나 전기적 고장 원인의 확률이 아닙니다.
검증된 촬영 조건에서 대상 결함이 있으면 **불량**, 없으면 **정상**이며,
크기·구도·노출이 범위 밖이면 **확인 필요**, 입력·추론 실패는 **검사 불가**입니다.
실제 촬영 성능은 아직 검증하지 않았습니다. 비교 기준과 결과는 [모델 비교 보고서](docs/native-comparison.md)에 있습니다.

개발 중 앱을 다시 빌드할 때는 저장소 루트의 학습 환경과 결과 파일을 사용합니다.

```bash
bash scripts/build_mac_inspector.sh
open "outputs/white-screen-inspector/White Screen Inspector.app"
```

빌드에는 Swift 컴파일러와 `artifacts/yolo-env`, `outputs/paired-model-comparison-v1/frozen_selection.json`,
선택 모델의 가중치가 필요합니다. 앱은 현재 프로젝트 폴더 안에서 실행하며 모델 해시를 확인합니다.
현재 지원 조건은 1024×1024 원본 타일 또는 고정 정면 구도의 5888×3368 전체 장면입니다.
전체 장면은 경계 위치도 확인합니다. 임의 사진을 자동 정렬하거나 축소해 정상으로 판정하지 않습니다.
앱의 ‘예시 사진’ 메뉴에서 정상과 각 결함·복합 사진을 바로 열 수 있습니다.
재빌드 후 macOS가 문서 폴더 접근을 다시 요청할 수 있습니다. 모델·설정을 읽기 위한 접근을
허용한 뒤 사진을 다시 선택하세요.

### 새 정상·불량 짝 데이터 v2

`data/oled-white-paired-v2/`는 정상 1,000장, 단일 불량 5종 각 1,000장, 복합 불량 1,000장으로
총 7,000장입니다. 작은 점·얇은 선·채널별 점진 저하·완만한 영역 저하를 포함합니다.
정상 원본 단위로 학습 4,900장·검증 1,050장·시험 1,050장을 분리합니다.

실제 저장한 학습 PNG는 원래 촬영 해상도를 유지한 1024×1024 타일입니다. 전체 장면은 고정 구도
5888×3368이며 원본 조건·좌표별 노이즈·결함을 기록해 필요할 때 정확히 내보냅니다.
학습 이미지의 경로와 라벨은 `dataset.yaml`, 짝 관계와 원본 조건은 `manifest.jsonl`,
`source_cases.jsonl`, 검증 결과는 `verification.json`에 있습니다. 검토 화면은 `review/index.html`입니다.
7,000장 모두에 독립 클래스 비트 마스크를 추가하고 두 모델을 같은 분할로 16 epoch 학습했습니다.
선택은 검증 자료만 사용하며, 시험은 축소하지 않은 전체 장면 1,050장으로 수행합니다.
U-Net과 YOLO의 공통 평가는 같은 클래스의 박스 IoU ≥ 0.5로 일대일 매칭합니다.
픽셀 Dice·IoU는 U-Net에 별도로 보고하며 YOLO 박스를 픽셀 마스크로 간주하지 않습니다.

```bash
.venv/bin/python scripts/generate_oled_full_dataset.py --output data/oled-white-paired-v2 --sources 1000
.venv/bin/python scripts/check_oled_full_dataset.py data/oled-white-paired-v2
artifacts/yolo-env/bin/python scripts/render_oled_full_review.py data/oled-white-paired-v2
.venv/bin/python scripts/export_oled_scene.py data/oled-white-paired-v2 panel0000_mixed --output outputs/exported-scene/panel0000_mixed.png
```

완료된 데이터는 덮어쓰지 않습니다. 실제 장비 보정 전 합성 자료이며,
크기·감소율·정상 편차는 생성 조건입니다. 더 자세한 저장·분할·라벨 기준은 데이터 생성 인계를 참고하세요.

### 픽셀 마스크와 모델 비교

마스크는 각 결함을 따로 렌더링했을 때 정상 대비 최대 채널 차이가 2 DN 이상인 픽셀입니다.
5개 클래스는 각각 비트 1·2·4·8·16을 사용해 겹침을 보존하며 정상 마스크는 0입니다.
합성 패널 1픽셀은 촬영 이미지 3픽셀에 대응합니다. 아래 ‘1픽셀 선’ 성적은 패널 기준입니다.

```bash
.venv/bin/python scripts/generate_oled_masks.py data/oled-white-paired-v2
PYTHONPATH=src artifacts/yolo-env/bin/python scripts/check_native_inference.py
PYTHONPATH=src artifacts/yolo-env/bin/python scripts/check_native_inspector.py
```

학습·검증·최종 시험 명령, 체크포인트와 자원 설정 변경, 전체 결과는
[비교 보고서](docs/native-comparison.md)와 `outputs/paired-model-comparison-v1/protocol.json`에 있습니다.
기존 데이터·모델·OpenCV 비교 스크립트는 보존했습니다. 아래 v1 결과는 이전 합성 자료의 기록입니다.

### 회색 화면 U-Net 실험

회색 48·96·144·192/255의 짝 사진과 마스크 3,360장을 만들고 흰 화면 모델을 5 epoch 미세 조정했습니다.
시험 504장에서 1패널픽셀 선 164개를 모두 찾았고, 정상 오탐은 0/72장이었습니다.
회색 48에서는 약한 영역 7개를 모두 놓쳤습니다. 회색별 표와 명령은
[회색 화면 비교 보고서](docs/gray-comparison.md)에 있습니다.

[평가 자료 ZIP](https://github.com/jangcuber/display-defect-ai/releases/download/v0.4.0/DisplayDefectAI-evaluation.zip)에는 흰 화면·회색 화면 모델과 설정, 시험 지표, 비교 그림이 있습니다.
[GitHub Releases](https://github.com/jangcuber/display-defect-ai/releases/tag/v0.4.0)에서 중간 발표 자료도 확인할 수 있습니다.

### 기존 OLED 5종 데이터 v1의 생성과 학습

기존 정상 1,000장의 촬영 구도와 원본 패널별 분할을 사용합니다. 정상과 5종 불량의 주 종류별로
1,000장씩 총 6,000장이고, 학습 4,200장·검증 900장·시험 900장입니다.
일부 불량 사진에는 군집점 또는 추가 결함이 들어가므로 실제 라벨 객체 수는 사진 수보다 많습니다.
군집점은 점별로, 끊긴 선은 한 선을 감싸는 박스로 라벨링합니다.

```bash
.venv/bin/python scripts/generate_oled_dataset.py --reference data/white-normal-fixed-v1 --output data/oled-white-five-v1
.venv/bin/python scripts/check_dataset.py data/oled-white-five-v1
YOLO_CONFIG_DIR="$PWD/artifacts/ultralytics" MPLCONFIGDIR="$PWD/artifacts/matplotlib" artifacts/yolo-env/bin/python scripts/train_oled.py --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-five-v1 --epochs 8
```

생성과 학습은 같은 출력 폴더를 덮어쓰지 않습니다. 학습에는 Apple MPS 접근이 필요합니다.
시험 데이터는 검증 분할에서 모델과 임계값을 고른 후 평가합니다.

```bash
artifacts/yolo-env/bin/python scripts/evaluate_oled.py collect --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-five-v1 --split val
artifacts/yolo-env/bin/python scripts/evaluate_oled.py calibrate --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-five-v1
artifacts/yolo-env/bin/python scripts/evaluate_oled.py collect --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-five-v1 --split test
artifacts/yolo-env/bin/python scripts/evaluate_oled.py test --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-five-v1
```

학습에는 1픽셀 선을 3회, 2픽셀 선을 2회 노출하는 가중 표본 목록을 사용합니다.
색·군집·끊김 속성은 검출 박스 내부의 영상에서 계산합니다. 크기의 단위는 512픽셀 정렬 영상의
픽셀이며 실제 패널의 서브픽셀 크기나 전기적 고장 원인을 뜻하지 않습니다.
흰 화면의 출력 저하를 모사한 합성 자료입니다. 휘점 검사는 검정 화면이 추가로 필요하며,
무라·번인·외관 이물 검사는 이 5종 모델의 평가 범위에 포함하지 않습니다.

### 기존 v1 자료의 얇은 선 검출 개선 기록

이전 앱은 기존 5종 가중치로 640픽셀 추론을 수행하고, 960픽셀 추론의 선 후보를 추가로 결합했습니다.
같은 방향·위치의 선 후보는 길이 방향으로 합치고 검증 분할에서 선택한 임계값과 중복 억제를 적용합니다.
YOLO 경로에는 OpenCV 검출 결과를 사용하지 않습니다. 추가 4 epoch 학습 후보도 비교했으나
검증 성능이 안정적으로 개선되지 않아 기존 가중치와 다중 해상도 추론을 채택했습니다.

같은 합성 시험 900장에서 1픽셀 선은 56/64개에서 64/64개, 전체 선은 356/365개에서
365/365개로 개선됐습니다. 정상 오탐 0/150장과 선 중복 박스 0개를 유지했습니다.
점·영역 검출 수는 유지됐으며 색 이상점 1개는 여전히 놓칩니다. 실제 촬영 성능은 미검증입니다.

재현 시 `outputs/yolo26n-oled-lines-v3/train/weights/best.pt`에 기존 5종 가중치를 준비한 뒤 실행합니다.
기존 결과는 보존하며 새 출력 폴더를 사용하세요.

```bash
artifacts/yolo-env/bin/python scripts/evaluate_oled.py collect --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-lines-v3 --size 640 --line-sizes 960 --split val
artifacts/yolo-env/bin/python scripts/evaluate_oled.py calibrate --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-lines-v3
artifacts/yolo-env/bin/python scripts/evaluate_oled.py collect --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-lines-v3 --size 640 --line-sizes 960 --split test
artifacts/yolo-env/bin/python scripts/evaluate_oled.py test --dataset data/oled-white-five-v1 --output outputs/yolo26n-oled-lines-v3
artifacts/yolo-env/bin/python scripts/compare_oled_runs.py --dataset data/oled-white-five-v1 --baseline outputs/yolo26n-oled-five-v1 --improved outputs/yolo26n-oled-lines-v3
```

비교 JSON과 전후 예시는 개선 결과 폴더의 `improvement_comparison.json`, `thin_line_examples.png`입니다.
시험 900장은 이전과 같은 자료를 재사용했으며 설정 선택에는 검증 900장만 사용했습니다.

## 합성 PGM 스모크 테스트

아래 스모크 테스트는 8×8 합성 PGM으로 입출력 연결을 확인합니다.
`SyntheticSmokePredictor`는 회색 기준값과의 차이를 계산하는 테스트 대역입니다.

Python 3.10 이상이 필요합니다. 저장소 루트에서 다음을 실행합니다. 외부 패키지나 데이터 다운로드는 없습니다.

```bash
python --version
python scripts/generate_synthetic.py
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/check.py
PYTHONPATH=src python -m display_defect_ai.cli tests/fixtures/synthetic/bright_patch.pgm
```

Windows PowerShell에서는 먼저 `$env:PYTHONPATH="src"`를 실행하고 각 명령의 `PYTHONPATH=src` 부분을 빼세요.
출력은 JSON입니다. `score`는 이상 정도이고 불량 확률이 아닙니다. `decision: "uncalibrated"`는
검증 데이터로 판정 기준을 정하지 않았다는 뜻입니다. 이 상태에서 화면에 정상/불량을 표시하면 안 됩니다.

선택: 이미 setuptools 68 이상과 pip가 있는 환경에서는 다음처럼 로컬 패키지를 설치할 수 있습니다.

```bash
python -m pip install --no-deps --no-build-isolation -e .
display-defect-smoke tests/fixtures/synthetic/flat.pgm
```

패키지 설치가 어려우면 위의 `PYTHONPATH` 방식만 사용해도 됩니다. 이 스모크 테스트에는 모델 패키지가 필요하지 않습니다.

`python scripts/check.py`는 핵심·독립 QA·헤드리스 데모 테스트를 함께 실행하며 건너뜀이나
예상 실패가 있으면 실패로 종료합니다. 로컬 데스크톱과 Tkinter가 있으면 `python demo/app.py`로
[합성 이미지 검사 데모](demo/README.md)를 엽니다. 클라우드에서는 GUI 시각 검증을 수행하지 못했습니다.

## 코드 읽는 순서

1. `scripts/generate_synthetic.py`: 사람이 작성한 작은 배열을 이미지 파일로 저장합니다.
2. `src/display_defect_ai/contracts.py`: 모델과 화면이 주고받는 데이터 약속입니다.
3. `src/display_defect_ai/synthetic.py`: 계약을 시험하는 단순 계산입니다.
4. `tests/`: 잘못된 입력, JSON 출력, 임계값 경계 등을 확인합니다.

학습 모델은 U-Net과 YOLO26n이며 로컬 모델과 실행 설정은 `outputs/`에 저장합니다.
현재 생성 데이터는 합성이며 실제 불량 검출 성능을 입증하지 않습니다.

## 함께 작업하기

- [인터페이스 v1](docs/interface.md): 데모/모델/QA가 공유하는 필드와 의미
- [역할과 세션 인계](docs/ownership.md): 충돌을 피하는 파일 담당 범위
- [한 달 계획](docs/plan.md): 데이터 확보 전후의 완료 기준
- [데이터·평가 규칙](docs/data.md): 출처, 분할, 누출 방지와 정직한 보고
- [1주차 학습 안내](docs/learning/week-01.md): 실행·경계값·설명 연습
- [독립 QA와 통합 결과](docs/learning/qa-review.md): 발견한 결함과 수정 검증

이 저장소는 PUBLIC입니다. 실제 이미지, 가중치, 개인 정보, 토큰은 커밋하지 마세요. 배포하는 합성 모델 가중치는 Releases에 둡니다.
`.gitignore`는 실수 방지용이며 모든 민감 파일을 감지하지는 못합니다. AI가 작성한 코드도 담당자가
입력/출력과 실패 조건을 설명하고 테스트 결과를 확인한 뒤 채택합니다.

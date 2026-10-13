# 로컬 이미지 검사 데모 (합성 전용)

**합성 테스트 / 실제 성능 미검증.** 실제 데이터·학습된 모델은 없습니다.
`SyntheticSmokePredictor`는 회색 0.5와의 차이를 계산하는 테스트 대역이며 AI 결함 탐지기가 아닙니다.
점수는 불량 확률이 아니며, 합성 결과로 성능을 주장하면 안 됩니다.

## 실행 (저장소 루트)

Python 3.10 이상과 데스크톱 화면, Tkinter가 필요합니다. 추가 pip 패키지, 웹 서버,
계정, 인터넷, GPU는 필요하지 않습니다. 핵심 패키지의 설치/설정도 바꾸지 않습니다.

```bash
python --version
python -m tkinter
python demo/app.py
```

Windows PowerShell에서도 같은 명령을 사용합니다 (`python` 대신 `py`가 필요할 수 있습니다).
`python -m tkinter`는 작은 확인 창을 엽니다. 닫고 데모를 실행하세요.
Tkinter가 없다면 Windows/macOS Python 설치 프로그램의 Tcl/Tk 옵션을 확인하세요.
Ubuntu/Debian에서는 운영체제 패키지 `python3-tk`가 필요합니다. 배포판 Python을 쓰면
`python3 demo/app.py`로 실행하세요. SSH/화면 없는 서버에서는 GUI가 실행되지 않습니다.
한글이 네모로 보이면 운영체제에 한글 글꼴을 설치하세요.

## 3분 시연

1. **합성 밝은 패치 예제**: 원본 8×8과 빨간 히트맵, 점수 1, `미보정 — 정상/불량 판정 없음`을 확인합니다.
2. **합성 평면 예제**: 점수 0으로 바뀌어도 여전히 미보정입니다. 정상이라고 판정하지 않습니다.
3. **합성 파일 선택**: 직접 만든 합성 PGM 또는 `tests/fixtures/synthetic/*.pgm`을 선택합니다.
4. **실제 추론 (미지원)**: 기존 결과가 사라지고 입력 버튼이 비활성화됩니다. 실제 추론은 호출하지 않습니다.
5. 합성 시연으로 돌아가 다시 실행합니다. 취소·입력 오류가 나면 이전 결과를 지우고 다시 선택할 수 있습니다.

파일 선택은 로컬 파일 읽기입니다. 이미지/결과를 디스크에 저장하거나 네트워크로 전송하지 않습니다.
결과 식별자는 고정된 `synthetic-upload`이며 파일명·전체 경로를 결과에 넣지 않습니다.
OS 파일 선택 창의 최근 파일 기록 등은 운영체제 설정에 따릅니다.

## 허용 파일과 제한

- ASCII `P2` 형식의 `.pgm`만 지원합니다. PNG/JPEG, 바이너리 PGM(P5), 실제 사진은 미지원입니다.
- 정확히 `# SYNTHETIC_SMOKE_ONLY`인 주석 줄이 필요합니다. 이 표시는 출처 검증 장치가 아니므로,
  실제 사진에 표시를 붙여 통과시키면 안 됩니다.
- 파일 1바이트–1 MiB, 가로/세로 각각 1–256픽셀, PGM 최댓값 1–65535입니다.
- 픽셀 수는 가로×세로와 같아야 하고 정수 값은 0–PGM 최댓값 범위여야 합니다.
- 읽기는 최대 1 MiB+1바이트로 제한하고 크기를 확인한 후 배열을 만듭니다.
  최댓값으로 나눈 회색조 [0,1]을 공유 `ImageInput`으로 전달합니다.

## 화면과 공유 계약

`inspection.py`는 `Predictor.predict(ImageInput) -> InferenceResult`를 호출합니다.
계약을 복사하지 않고 `src/display_defect_ai/contracts.py`에서 import합니다.
원본 식별자·크기와 합성 출처가 출력과 같은지도 확인합니다.
[인터페이스 v1](../docs/interface.md)을 기준으로 유지하세요.

- `score`: 유한한 이상 정도. UI가 확률로 바꾸거나 임계값을 만들지 않습니다.
- `calibration=None`: 항상 `uncalibrated`, 정상/불량 판정 없음. 현재 시연은 전부 이 상태입니다.
- 보정된 어댑터는 계약의 `decision`과 `threshold`를 그대로 표시하며 보정 출처를 함께 보여 줍니다.
- 히트맵: 원본 height×width, 좌상단 원점. 정수 배율로만 화면 확대하고 이미지별 재정규화는 하지 않습니다.
  null이면 `위치 정보 없음`을 표시합니다. 빨강은 표시값의 크기이며 결함 확률이 아닙니다.
- 아래 JSON 영역은 히트맵 배열을 제외한 메타데이터입니다. 모델/버전/전처리/학습 실행/합성 여부와 보정 출처를 표시합니다.
  현재 `training_run_id=null`은 학습하지 않았다는 뜻입니다.

실제 모델 연결은 별도 작업입니다. 리더가 데이터 사용 권한, 실제 이미지 디코딩/전처리와 크기 제한,
학습 기록, 검증 분할에 의한 보정을 확정한 뒤 어댑터를 제공해야 합니다.
대역만 바꾸고 실제 모드를 활성화하지 마세요. 현재 앱은 실제 입력을 의도적으로 거부합니다.

## 검증 및 초보 팀원 인계

저장소 루트에서 실행합니다. GUI 없이도 동작합니다.

```bash
PYTHONPATH=src python -m unittest discover -s demo -v
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m display_defect_ai.cli tests/fixtures/synthetic/bright_patch.pgm
```

PowerShell에서는 먼저 `$env:PYTHONPATH="src"`를 실행하고 `PYTHONPATH=src` 부분을 빼세요.
데모 테스트는 입력 제한 경계, 잘못된 형식/값/크기/인코딩, 파일 접근 오류, 반복 실행,
오류 후 복구, 실제 모드 호출 차단, 출처/크기 불일치, 선택 취소용 상태 초기화,
선택적 히트맵/보정 보존을 확인합니다. 테스트의 보정값은 분기 확인용이며 실제 보정 결과가 아닙니다.

GUI 확인은 위 3분 시연에 더해 잘못된 파일 선택, 취소, 긴 창/작은 창, 재실행을 확인하세요.
자동 테스트만으로 OS별 파일 선택 창·한글 글꼴·화면 배치를 검증할 수는 없습니다.
읽는 순서는 `inspection.py` → `test_demo.py` → `app.py`입니다.
초보 팀원의 한 번의 작업은 한 가지 안내 문구 또는 한 가지 실패 흐름 정도로 제한하고,
공유 인터페이스나 의존성 변경은 통합 담당에게 먼저 알리세요.

## 통합

이 변경은 foundation `281eb3b3560ac8ad92f90c6488576a0d9073b2cb` 위의 `review/demo`입니다.
PR 기준 브랜치는 `review/project-foundation`이며 변경 범위는 `demo/`뿐입니다.
foundation 통합 후 리더가 데모 브랜치를 새 기준에 rebase하거나 데모 커밋만 cherry-pick하고,
위 두 테스트 묶음과 GUI 확인을 다시 실행하세요. 데이터·가중치·개인정보는 커밋하지 마세요.

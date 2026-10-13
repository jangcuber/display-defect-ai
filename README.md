# Display Defect AI — 학습 프로젝트

디스플레이 사진 한 장을 입력받아 이상 점수와 위치 히트맵을 보여 주는 작은 데모를 목표로 합니다.
5명이 시험 기간을 제외한 최대 8주 동안 진행합니다. 리더는 주 최대 30시간,
초보 팀원 4명은 각 주 약 3시간을 기준으로 범위를 줄였습니다.

**현재 실제 데이터, 학습된 모델, 성능 측정 결과는 없습니다.** 포함된 이미지는 코드로 만든
8×8 합성 PGM입니다. 합성 스모크 테스트는 입출력 연결만 확인하며, 실제 결함 탐지 성능을 증명하지 않습니다.
`SyntheticSmokePredictor`는 회색 기준값과의 차이를 계산하는 테스트 대역이며 AI 모델이나 PatchCore가 아닙니다.

## 바로 실행하기

Python 3.10 이상이 필요합니다. 저장소 루트에서 다음을 실행합니다. 외부 패키지나 데이터 다운로드는 없습니다.

```bash
python --version
python scripts/generate_synthetic.py
PYTHONPATH=src python -m unittest discover -s tests -v
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

패키지 설치가 어려우면 위의 `PYTHONPATH` 방식만 사용해도 됩니다. 모델 관련 대형 패키지는 아직 설치하지 않습니다.

## 코드 읽는 순서

1. `scripts/generate_synthetic.py`: 사람이 작성한 작은 배열을 이미지 파일로 저장합니다.
2. `src/display_defect_ai/contracts.py`: 모델과 화면이 주고받는 데이터 약속입니다.
3. `src/display_defect_ai/synthetic.py`: 계약을 시험하는 단순 계산입니다.
4. `tests/`: 잘못된 입력, JSON 출력, 임계값 경계 등을 확인합니다.

실제 정상 사진만으로 기준을 학습하는 기존 이상 탐지 방법(예: PatchCore)을 다음 단계의 후보로 둡니다.
데이터 사용 권한·종류·수량과 CPU/GPU·메모리를 확인한 다음 구현체와 버전을 선택합니다.
직접 논문 모델을 처음부터 구현하거나 여러 모델의 성능 경쟁을 하는 것은 이번 범위에 포함하지 않습니다.

## 함께 작업하기

- [인터페이스 v1](docs/interface.md): 데모/모델/QA가 공유하는 필드와 의미
- [역할과 세션 인계](docs/ownership.md): 충돌을 피하는 파일 담당 범위
- [8주 계획](docs/plan.md): 데이터 확보 전후의 완료 기준
- [데이터·평가 규칙](docs/data.md): 출처, 분할, 누출 방지와 정직한 보고

이 저장소는 PUBLIC입니다. 실제 이미지, 가중치, 개인 정보, 토큰은 커밋하지 마세요.
`.gitignore`는 실수 방지용이며 모든 민감 파일을 감지하지는 못합니다. AI가 작성한 코드도 담당자가
입력/출력과 실패 조건을 설명하고 테스트 결과를 확인한 뒤 채택합니다.

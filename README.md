# Display Defect AI — 학습 프로젝트

한 달 안에 혼자 데이터 생성 → YOLO26n 학습 → 맥 검사 → 평가를 한 번 완성한 뒤 팀원용 템플릿을 만듭니다.
첫 대상은 점·선이며 OpenCV 전처리/규칙 기반 비교 기준과 지도학습 객체 검출을 사용합니다.
사진별 검출은 별도 프로그램 규칙으로 종합합니다. 무라 전용 모델은 기본 과정 이후 판단합니다.

학습용 정상·점·선 PNG와 YOLO 라벨 생성기는 [데이터 생성 인계](docs/generation-handoff.md)를 참고하세요.
기존 8×8 PGM 데모는 계약 테스트 대역으로 유지하며 새 학습 데이터와 구분합니다.

**현재 실제 데이터, 학습된 모델, 성능 측정 결과는 없습니다.** 포함된 이미지는 코드로 만든
8×8 합성 PGM입니다. 합성 스모크 테스트는 입출력 연결만 확인하며, 실제 결함 탐지 성능을 증명하지 않습니다.
`SyntheticSmokePredictor`는 회색 기준값과의 차이를 계산하는 테스트 대역이며 AI 모델이나 PatchCore가 아닙니다.

## 바로 실행하기

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

패키지 설치가 어려우면 위의 `PYTHONPATH` 방식만 사용해도 됩니다. 모델 관련 대형 패키지는 아직 설치하지 않습니다.

`python scripts/check.py`는 핵심·독립 QA·헤드리스 데모 테스트를 함께 실행하며 건너뜀이나
예상 실패가 있으면 실패로 종료합니다. 로컬 데스크톱과 Tkinter가 있으면 `python demo/app.py`로
[합성 이미지 검사 데모](demo/README.md)를 엽니다. 클라우드에서는 GUI 시각 검증을 수행하지 못했습니다.

## 코드 읽는 순서

1. `scripts/generate_synthetic.py`: 사람이 작성한 작은 배열을 이미지 파일로 저장합니다.
2. `src/display_defect_ai/contracts.py`: 모델과 화면이 주고받는 데이터 약속입니다.
3. `src/display_defect_ai/synthetic.py`: 계약을 시험하는 단순 계산입니다.
4. `tests/`: 잘못된 입력, JSON 출력, 임계값 경계 등을 확인합니다.

학습 모델은 YOLO26n을 첫 대상으로 정했습니다. 모델 설치·가중치·학습은 다음 세션에서 검증합니다.
현재 생성 데이터는 합성이며 실제 불량 검출 성능을 입증하지 않습니다.

## 함께 작업하기

- [인터페이스 v1](docs/interface.md): 데모/모델/QA가 공유하는 필드와 의미
- [역할과 세션 인계](docs/ownership.md): 충돌을 피하는 파일 담당 범위
- [한 달 계획](docs/plan.md): 데이터 확보 전후의 완료 기준
- [데이터·평가 규칙](docs/data.md): 출처, 분할, 누출 방지와 정직한 보고
- [1주차 학습 안내](docs/learning/week-01.md): 실행·경계값·설명 연습
- [독립 QA와 통합 결과](docs/learning/qa-review.md): 발견한 결함과 수정 검증

이 저장소는 PUBLIC입니다. 실제 이미지, 가중치, 개인 정보, 토큰은 커밋하지 마세요.
`.gitignore`는 실수 방지용이며 모든 민감 파일을 감지하지는 못합니다. AI가 작성한 코드도 담당자가
입력/출력과 실패 조건을 설명하고 테스트 결과를 확인한 뒤 채택합니다.

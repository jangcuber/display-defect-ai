# Foundation 독립 QA 인계

검토 기준: `281eb3b3560ac8ad92f90c6488576a0d9073b2cb` (`review/project-foundation`).
QA 브랜치: `review/qa-learning`. Python 3.12.14, 외부 의존성 없는 unittest 환경.
`docs/interface.md`, `docs/ownership.md`, `docs/data.md`, `docs/plan.md`와 전체 소스/기존
테스트를 읽었습니다. 저장소 및 작업공간의 `AGENTS.md`, `.agents/skills`는 없었습니다.
변경은 새 `tests/test_qa_edges.py`와 `docs/learning/`에만 있습니다.

## 재현 결과

| 검사 | 결과 |
|---|---|
| 변경 전 `PYTHONPATH=src python -m unittest discover -s tests -v` | 기존 11개 통과, 실패 0 |
| 변경 후 같은 명령 | 총 26개: 통과 22, 예상 실패 4, 일반 실패/오류 0, 건너뜀 0 |
| 예상 실패 표시를 해제한 결함 재현 | 총 4개: assertion 실패 3, AttributeError 오류 1 |
| `python scripts/generate_synthetic.py` 후 fixture diff | 변경 없음 |
| flat / bright_patch CLI JSON | 각각 score 0 / 1, 모두 uncalibrated, 합성 출처 및 미학습 표시 확인 |
| `git diff --check` | 통과 |
| Python 3.10/3.11 및 Windows | 미실행 |
| editable 설치 / 패키징 / 실제 이미지 / 모델 학습 / 성능 평가 | 미실행 |

추가 15개 테스트 메서드는 NaN/±inf/비수치 입력, bool과 숫자 구별, 직사각형 좌표·행/열,
정규화 범위와 score 범위 구별, 인접 부동소수점 threshold 경계, JSON 판정/보정 일치,
필수 출처, 반복 실행, 합성 전용 입력, PGM 한계값과 marker를 검사합니다.
subTest 사례 수를 독립 테스트 개수로 부풀리지 않았습니다.

예상 실패는 수정 전 결함을 드러내면서 정상 회귀 검사를 실행하기 위한 표시입니다.
`OK (expected failures=4)`를 “26개 통과”로 보고하지 마세요. core 수정 후 해당 decorator를
제거하고 다시 실행합니다. 그 전에 수정이 들어오면 unexpected success가 되어 확인을 요구합니다.

## core 담당자에게 전달할 결함

| ID / 심각도 | 위치와 재현 | 영향 / 요청 |
|---|---|---|
| QA-01 / P2 보통 | `InferenceResult.__post_init__`: `InferenceResult("qa", 1, 1, 0.5, None, None)`이 생성되고 `to_dict()`에 provenance null 출력 | 필수 출처 계약을 벗어난 JSON이 유통될 수 있음. Provenance 객체 검증 요청. None뿐 아니라 dict 등 잘못된 타입도 검토 |
| QA-02 / P2 보통 | 같은 생성자에 calibration `{"threshold": 0.5}` 전달 시 생성 성공, `decision`/`to_dict()` 접근 시 AttributeError | 생성 경계를 통과한 뒤 소비자가 실패함. None 또는 Calibration 검증 요청 |
| QA-03 / P3 낮음 | `ImageInput(None, 1, 1, ((0.5,),), True)`가 ValueError 대신 AttributeError | 문서의 잘못된 입력 오류 규약 불일치. 문자열 필드의 타입 검사 및 grid 구조 오류도 함께 검토 요청 |
| QA-04 / P2 보통 (통합 합의 반영 대기) | score=threshold=0.5에서 foundation은 anomaly 반환 | 합의된 규칙은 `score > threshold`만 anomaly, 동점은 normal. core 담당자가 구현·계약·기존 테스트를 함께 수정 예정 |

2026-10-13 통합 지시에 따라 새 QA는 엄격한 `>` 경계를 기대합니다. 재확인한 원격 foundation
`4015c09096b3556ca4b5181b3cc8c40f60c5e467`에는 데이터 문서만 추가되어 아직 `>=` 구현입니다.
QA 브랜치는 요청한 정확한 원본 커밋을 부모로 유지합니다. 인접 양쪽 값 검사는 통과하며
동점 검사는 별도 예상 실패입니다. 기존 `tests/test_contracts.py`의 동점 기대값은 소유 범위 밖이므로
수정하지 않았습니다. core 변경을 통합한 뒤 기존 테스트·문서·QA가 함께 `>`를 사용하는지 확인하세요.

현재 CLI 정상 경로는 올바른 객체를 직접 만들므로 QA-01~03의 직접 영향은 어댑터 통합 경계입니다.
핵심 코드나 데모는 수정하지 않았습니다. 각 항목은 `QAKnownContractDefects`의 작은 테스트로 재현합니다.
예상 실패를 숨기지 않고 실패/오류 자체를 확인하는 명령:

```bash
PYTHONPATH=src:tests python - <<'PY'
import unittest
from test_qa_edges import QAKnownContractDefects
for name in unittest.defaultTestLoader.getTestCaseNames(QAKnownContractDefects):
    getattr(QAKnownContractDefects, name).__unittest_expecting_failure__ = False
suite = unittest.defaultTestLoader.loadTestsFromTestCase(QAKnownContractDefects)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

## 보호장치의 한계

명시적 `is_synthetic=False`와 marker 없는 PGM은 거부됩니다. 하지만 marker와 boolean은
호출자가 제공하는 선언이지 이미지의 실제 출처를 인증하는 장치가 아닙니다. 실제 이미지를
합성이라고 다시 표시해 이 대역에 넣지 마세요. 출처 권한과 분할은 별도 기록으로 확인합니다.
출처 필드가 있다는 사실만으로 유효한 학습·보정·누출 방지가 증명되지는 않습니다.

현재 실제 데이터·학습된 탐지기·실제 성능 측정은 없습니다. score/heatmap/단위 테스트 threshold는
합성 연결 검사에만 사용합니다. test로 전처리나 threshold를 선택하지 않고, 데이터 확보 이후
그룹 분할·실행 버전/시드/설정·학습 및 보정 출처를 고정해야 합니다. AI 작성 코드도 같은 기준을 적용합니다.

## 통합 순서

1. foundation 기준 stacked draft PR에서 두 학습 문서와 새 QA 파일만 검토합니다.
2. core 담당자는 QA-01~04를 자신의 범위에서 수정하고 해당 재현을 실행합니다.
3. QA 담당자는 고친 항목의 expectedFailure를 제거해 실제 통과를 확인합니다.
4. 통합 담당자는 foundation 위에 QA 커밋을 cherry-pick하거나 승인된 PR 절차를 사용합니다.
   자동 merge는 하지 않습니다. 다른 기반에 적용하면 전체 테스트와 CLI를 다시 실행합니다.

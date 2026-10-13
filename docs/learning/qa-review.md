# Foundation 독립 QA와 통합 결과

초기 검토 기준은 `281eb3b3560ac8ad92f90c6488576a0d9073b2cb`입니다.
독립 QA의 [초안 PR #2](https://github.com/jangcuber/display-defect-ai/pull/2)와
데모의 [초안 PR #3](https://github.com/jangcuber/display-defect-ai/pull/3)는 별도 검토 기록입니다.
QA 구현 커밋 `faedcec33b5140b7c1c953d28562e5413815d91a`, `19fa296c1aee025e00a1d7e4484c178fea850bb5`와
데모 구현 커밋 `216f477777b966f632928aef05bc382aff34185d`, `46e7e5dc66a9ea2fa49ad6e551b7ef7929a8130d`를
통합 리뷰 브랜치에 cherry-pick했습니다. 이후 브랜치의 merge 커밋은 가져오지 않았으며 main에 병합하지 않았습니다.

## 현재 검증 결과

Python 3.12.14에서 `python scripts/check.py`로 **39개 모두 통과**했습니다.
핵심·독립 QA 31개와 헤드리스 데모 8개입니다. 일반 실패, 오류, 예상 실패, 예상 밖 성공, 건너뜀은 모두 0개입니다.
subTest 사례 수는 독립 테스트 수로 부풀리지 않습니다. 이 통합 실행기는 건너뜀이나 예상 실패도 성공으로 보고하지 않습니다.

```bash
python scripts/check.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s demo -v
```

PowerShell에서는 개별 명령 전에 `$env:PYTHONPATH="src"`를 설정하고 `PYTHONPATH=src` 부분을 빼세요.
통합 실행기는 별도 PYTHONPATH 설정 없이 동작합니다.

## 발견한 결함과 수정

| ID / 초기 심각도 | 재현된 문제 | 통합 수정과 회귀 검사 |
|---|---|---|
| QA-01 / P2 | provenance=None이 생성되어 필수 출처 없는 JSON 출력 | Provenance 객체를 생성 시 검증. None·dict·문자열·숫자·bool 거부 |
| QA-02 / P2 | dict calibration이 생성된 뒤 decision에서 AttributeError | Calibration 객체 또는 None만 허용하고 생성 시 ValueError |
| QA-03 / P3 | ImageInput(None, ...)이 ValueError 대신 AttributeError | 모든 필수 문자열 타입 검증, 잘못된 grid 구조와 수치도 ValueError |
| QA-04 / P2 | 초기 >= 규칙에서 score=threshold를 anomaly로 분류 | 합의된 엄격한 >로 통일. 동점은 normal, 인접 부동소수점과 JSON 경계 확인 |

기존 결함 재현 테스트 네 개의 expectedFailure 표시를 제거했습니다. 누락 출처·보정 dict·잘못된 ID는
수정 전 다시 실행해 assertion 실패 2개와 AttributeError 1개를 재현했습니다. 비교 규칙은 이미 수정되어
그 시점에 동점 테스트는 통과했습니다. 수정 후 모든 재현이 실제 통과합니다.

추가로 문자열 필드 전체의 잘못된 타입, grid 컨테이너, 중첩 객체, float 변환 범위를 넘는 정수를 검사합니다.
잘못된 타입을 뒤늦게 소비자에서 실패시키지 않고 생성 경계에서 ValueError로 거부합니다.

## 독립 QA의 초기 결과 기록

원본 foundation 테스트는 11개 통과했습니다. comparator 변경 전 QA 보고의 총 26개는
**22개 통과와 4개 예상 실패**였으며, 26개 통과를 뜻하지 않습니다. 그 당시의 설치·GUI·실데이터 검사는
실행되지 않았습니다. 이 과거 결과를 현재 통합 상태와 혼동하지 마세요.

## 추가 검증과 남은 한계

통합 후 compileall, fixture 재생성 일치, whitespace, 오프라인 editable 설치, 설치된 CLI의 두 fixture,
문서 내부 링크와 ignore 규칙 검사를 모두 통과했습니다. GUI는 화면 없는 환경에서 안내 메시지를 출력하고
종료 코드 1로 끝나는 것을 확인했으며 traceback은 없었습니다. 최종 PR에도 실행 결과를 기록합니다.
Python 3.10/3.11과 Windows에서는 실행하지 않았습니다. GUI 시각 검증은 헤드리스 클라우드에서
불가능했으며 데스크톱의 한글 글꼴·레이아웃·파일 선택·클릭 흐름을 별도로 확인해야 합니다.

명시적 is_synthetic=False와 marker 없는 PGM은 거부되지만 marker와 boolean은 호출자의 선언이지
실제 출처를 인증하는 장치가 아닙니다. 실제 이미지에 합성 표시를 붙여 넣지 마세요.
출처 필드의 존재만으로 학습·보정·누출 방지가 증명되지는 않습니다.

현재 실제 데이터·학습된 탐지기·실제 성능 측정은 없습니다. score·heatmap·테스트 threshold는
합성 연결 검사에만 사용합니다. test로 전처리나 threshold를 선택하지 않고, 데이터 확보 후
그룹 분할·버전·시드·학습 및 보정 출처를 고정해야 합니다. AI 작성 코드도 같은 기준을 적용합니다.

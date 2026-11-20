# 합성 스모크 fixture 전용

`python scripts/generate_synthetic.py`로 생성한 8×8 ASCII PGM입니다.
실제 패널 사진이나 결함 데이터가 아닙니다. 무작위 요소와 외부 출처가 없습니다.

- `flat.pgm`: 모든 값이 50/100인 회색 배열
- `bright_patch.pgm`: 중앙 2×2만 100/100으로 바꾼 배열

점수 차이와 좌표/JSON 연결을 시험합니다. 두 fixture의 점수 분리는 실제 결함 탐지 성능의 근거가 아닙니다.

# White Screen Inspector for macOS

GitHub Releases의 `WhiteScreenInspector-macos-arm64.zip`을 내려받아 풀고
`White Screen Inspector.app`을 엽니다. **Apple Silicon 및 macOS 15 이상**용입니다.
앱 안에 오프라인 검사 런타임, 흰 화면 U-Net 체크포인트와 합성 예시 7장이 들어 있습니다.
**예시 사진 → 복합 불량**을 열면 점·선·영역을 확인할 수 있습니다.

이 버전은 합성 데이터 검사용 개발 빌드입니다. 검증된 입력은 1024×1024 타일과
고정된 구도의 5888×3368 흰 화면 장면입니다. 회색 실험 결과와 모델 가중치는 별도
평가 자료로 제공하며 이 앱의 판정은 흰 화면 기준입니다.
Apple Developer ID 공증을 받은 패키지가 아니므로 macOS가 앱 열기를 제한할 수 있습니다.
그 경우 아래의 소스 빌드 경로를 사용할 수 있습니다.

## 소스에서 빌드

Apple Silicon Mac과 Swift 컴파일러, macOS 15 이하를 지원하는 **arm64 Python 3.12**가 필요합니다.
`data/oled-white-paired-v2/`는 README의 생성 명령으로 준비합니다.
배포 평가 자료의 흰 화면 U-Net `best.pt` 및 `frozen_selection.json`은
`outputs/paired-model-comparison-v1/`에 평가 때와 같은 디렉터리 구조로 둡니다.

```bash
python3.12 -m venv artifacts/release-mac-env
artifacts/release-mac-env/bin/python -m pip install -r requirements/release-mac.txt
bash scripts/build_mac_release.sh
```

선택된 모델·코드·예시 파일의 해시를 빌드 때 확인합니다. ZIP은 `outputs/mac-release/`에 생성됩니다.

## 포함된 제삼자 구성 요소의 소스

번들의 `Contents/Resources/Third Party Notices/`에 설치된 패키지 버전과 라이선스 고지를 보관했습니다.

- [CPython](https://github.com/python/cpython/tree/3.12) 및 [Python Build Standalone](https://github.com/astral-sh/python-build-standalone)
- [PyTorch](https://github.com/pytorch/pytorch/tree/v2.15.0), [NumPy](https://github.com/numpy/numpy/tree/v2.5.4), [Pillow](https://github.com/python-pillow/Pillow/tree/13.0.0)
- [OpenCV Python 휠 빌드 소스](https://github.com/opencv/opencv-python), [OpenCV](https://github.com/opencv/opencv) 및 [FFmpeg](https://ffmpeg.org/download.html)
- [PyInstaller](https://github.com/pyinstaller/pyinstaller/tree/v6.22.3)

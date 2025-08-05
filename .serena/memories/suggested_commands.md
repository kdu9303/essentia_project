# Suggested Commands for Essentia Audio Analysis Project

## Development Environment Setup
```bash
# 의존성 설치
uv sync

# 가상환경 활성화 후 실행
uv run main.py
```

## Running the Project
```bash
# 메인 스크립트 실행
uv run main.py

# Python 직접 실행 (가상환경 내에서)
python main.py
```

## Model and Audio Processing
- 모델 파일들은 `models/` 디렉토리에 JSON 형태로 저장
- 오디오 파일 처리는 16kHz 샘플레이트 사용
- AudioClassifier를 통한 다양한 음악 특성 예측 가능

## Package Management
- **Package Manager**: uv 사용
- **Requirements**: pyproject.toml에 정의됨
- **Lock File**: uv.lock으로 정확한 버전 관리

## System Utilities (macOS/Darwin)
- `ls`, `cd`, `grep`, `find` 등 표준 Unix 명령어 사용 가능
- `git` 명령어로 버전 관리
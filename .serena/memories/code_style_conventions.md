# Code Style and Conventions

## 관찰된 코드 스타일
- **언어**: Python ≥3.8
- **변수명**: snake_case 사용 (audio_files, models_dir)
- **클래스명**: PascalCase 사용 (AudioClassifier, FeatureExtractor)
- **메서드명**: snake_case 사용 (predict_danceability, _common_predict)
- **비공개 메서드**: underscore prefix 사용 (_dowonload_models, _select_embeddings)

## 코드 구조 패턴
- **클래스 기반 설계**: AudioClassifier에 25개 예측 메서드 집중
- **유틸리티 분리**: utils.py에 독립적인 헬퍼 함수들
- **모듈화**: 기능별로 별도 파일 분리 (classifiers, feature_extractors, utils)

## 프로젝트 특성
- **다국어 지원**: 한국어 출력 및 주석 사용 (CLAUDE.md에 "always answer in Korean" 규칙)
- **오디오 처리 중심**: Essentia 라이브러리 활용한 음악 분석
- **모델 기반 예측**: 사전 훈련된 TensorFlow 모델 활용

## 아키텍처 패턴
- **팩토리 패턴**: 모델 선택 및 로딩
- **전략 패턴**: 다양한 예측 메서드들
- **유틸리티 패턴**: 공통 기능 분리
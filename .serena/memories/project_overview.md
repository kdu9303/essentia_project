# Essentia Audio Analysis Project Overview

## Purpose
Essentia 기반 음악 자동 태깅, 분류 및 임베딩 추출을 위한 TensorFlow 모델 활용 프로젝트

## Tech Stack
- **Python**: ≥3.8
- **Package Manager**: uv (uv.lock 파일 존재)
- **Core Dependencies**:
  - essentia>=2.1b6.dev1177 (오디오 분석)
  - essentia-tensorflow>=2.1b6.dev1177 (TensorFlow 통합)
  - matplotlib>=3.7.5 (시각화)
  - pandas>=2.0.3 (데이터 처리)
  - polars[pyarrow]>=1.8.2 (고성능 데이터 처리)
  - rich>=14.1.0 (CLI 출력)
  - ffmpeg-python>=0.2.0 (오디오 처리)

## Project Structure
- `models/`: 사전 훈련된 TensorFlow 모델들 (22개 .json 파일)
  - MusiCNN, VGGish 기반 다양한 분류/예측 모델
- `main.py`: 메인 실행 파일
- `classifiers.py`: AudioClassifier 클래스 (25개 예측 메서드)
- `utils.py`: 오디오 파일 처리 유틸리티
- `feature_extractors.py`: FeatureExtractor 클래스

## Core Architecture
- AudioClassifier: 중앙 분류 클래스
  - 25개 예측 메서드 (감정, 댄스 가능성, 악기 분류 등)
  - 모델 다운로드 및 메타데이터 관리
- FeatureExtractor: 특징 추출 클래스
- 유틸리티 함수들: 오디오 파일 처리, 자르기 등
# Essentia Audio Analysis Project

## 프로젝트 개요
TensorFlow 모델을 활용한 음악 자동 태깅, 분류 및 임베딩 추출에 중점을 둔 Essentia 오디오 분석 프로젝트입니다.

## 주요 기능
- **자동 태깅**: MusiCNN 모델을 사용한 음악 태그 자동 생성
- **분류**: VGGish 기반 모델을 통한 감정(기쁨/슬픔) 및 댄스 가능성 분류
- **임베딩 추출**: 다운스트림 작업을 위한 오디오 특징 벡터 추출

## 환경 요구사항
- Python ≥3.8
- 주요 의존성: essentia, essentia-tensorflow, matplotlib
- 패키지 관리자: uv

## 사용법
```bash
uv sync

uv run main.py
```

## 모델
- `models/` 디렉토리에 사전 훈련된 TensorFlow 모델 포함
- MusiCNN 및 VGGish 기반의 다양한 분석 모델 제공
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

## AudioClassifier 클래스

`AudioClassifier`는 음악 분석을 위한 핵심 클래스로, 다양한 예측 함수를 제공합니다.

### 기본 예측 함수
- `predict_approachability()` - 친근감/접근성 측정
- `predict_engagement()` - 몰입도/참여도 분석  
- `predict_danceability()` - 댄스 가능성 평가

### 무드/감정 예측 함수
- `predict_aggressive()` - 공격성/격렬함 정도
- `predict_happy()` - 행복감/기쁨 수준
- `predict_party()` - 파티 분위기/축제감
- `predict_relaxed()` - 편안함/릴랙스 정도
- `predict_sad()` - 슬픔/우울함 수준
- `predict_mirex()` - MIREX 감정 분류 (학술 표준)
- `predict_jamendo_mood_and_theme()` - Jamendo 무드/테마 분류

### 악기/음향 예측 함수
- `predict_acoustic()` - 어쿠스틱 악기 사용도
- `predict_electronic()` - 일렉트로닉 사운드 정도
- `predict_voice_instrumental()` - 보컬/악기 구분
- `predict_gender()` - 보컬 성별 분류
- `predict_timber()` - 음색/텍스처 분석
- `predict_tonality()` - 조성/무조성 분류

### 고급 음향 분석 함수
- `predict_nsynth_acoustic_electronic()` - NSynth 어쿠스틱/일렉트로닉 분류
- `predict_nsynth_bright_dark()` - NSynth 밝음/어두움 분석
- `predict_nsynth_reverb()` - NSynth 리버브 탐지
- `predict_arousal_valence_deam()` - DEAM 각성도/감정가 (연속값)
- `predict_arousal_valence_muse()` - MuSe 각성도/감정가 (멀티모달)
- `predict_tempo()` - 템포/비트 분석
- `predict_dynamic_complexity_loudness()` - 동적 복잡도/라우드니스 측정
- `predict_dissonance()` - 불협화도/협화도 분석


### 사용 예시
```python
from classifiers import AudioClassifier

# 분류기 초기화
classifier = AudioClassifier("audio_file.mp3")

# 개별 분석
danceability = classifier.predict_danceability()
mood = classifier.predict_happy()

# 전체 분석 (권장)
all_results = classifier.predict_all()

# 특정 함수 제외하고 분석
fast_results = classifier.predict_all(
    exclude_methods=['predict_tempo', 'predict_dissonance']
)
```
# Essentia Audio Analysis Project

## 프로젝트 개요
TensorFlow 모델을 활용한 음악 자동 태깅, 분류 및 임베딩 추출에 중점을 둔 Essentia 오디오 분석 프로젝트입니다. Essentia의 TensorFlow 통합을 사용하여 오디오 특징을 분석하고 음악적 특성에 대한 예측을 수행합니다.

## 주요 기능
- **자동 태깅**: MusiCNN 모델을 사용한 음악 태그 자동 생성
- **분류**: VGGish 기반 모델을 통한 감정(기쁨/슬픔) 및 댄스 가능성 분류
- **임베딩 추출**: 다운스트림 작업을 위한 오디오 특징 벡터 추출

## 환경 요구사항 및 설치

- **Python**: ≥3.8 required
- **패키지 관리자**: `uv` (uv.lock 파일 포함)
- **핵심 의존성**:
  - `essentia>=2.1b6.dev1177` 및 `essentia-tensorflow>=2.1b6.dev1177` (핵심 오디오 분석)
  - `matplotlib>=3.7.5` (시각화)
  - `polars[pyarrow]>=1.8.2` (고성능 데이터 처리)
  - `rich>=14.1.0` (CLI 출력)
  - `ffmpeg-python>=0.2.0` (오디오 처리)

```bash
# 환경 설정 및 실행
uv sync                    # 의존성 설치
uv run main.py            # 메인 스크립트 실행
```

## 프로젝트 구조

- `models/`: 다양한 오디오 분석 작업을 위한 사전 훈련된 TensorFlow 모델 (22개 JSON 파일)
  - 무드 분류: `mood_happy`, `mood_sad`, `mood_relaxed`, `mood_party` 등
  - 오디오 특성: `danceability`, `voice_instrumental`, `gender`, `timbre`
  - 음악 분석: `msd-musicnn-1`, `moods_mirex`, `mtg_jamendo_moodtheme`
- `main.py`: 오디오 처리 파이프라인이 포함된 메인 실행 스크립트
- `classifiers.py`: AudioClassifier 클래스 (25개 이상의 예측 메소드)
- `utils.py`: 오디오 파일 처리 유틸리티 (로딩, 커팅)
- `feature_extractors.py`: 오디오 특징 추출을 위한 FeatureExtractor 클래스

## AudioClassifier 예측 함수

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

## 사용 예시

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

## 핵심 아키텍처

**AudioClassifier**: 전문적인 예측 메소드를 가진 중앙 분류 클래스
- 모델 관리: 자동 다운로드 및 메타데이터 처리
- 25개 이상의 예측 메소드로 감정, 음악 특성, 오디오 특징 커버
- 공통 예측 파이프라인: 일관된 처리를 위한 `_common_predict()` 메소드

**주요 Essentia 컴포넌트**:
- `TensorflowPredictMusiCNN`: MusiCNN 기반 모델용 (자동 태깅)
- `TensorflowPredictVGGish`: VGGish 기반 모델용 (분류, 임베딩)
- `TensorflowPredict2D`: 2D 예측 모델용
- `MonoLoader`: 구성 가능한 샘플 레이트 (일반적으로 16kHz)로 오디오 로딩

## 모델 사용 패턴

1. **자동 태깅**: 오디오 로드 → MusiCNN으로 특징 추출 → 태그 활성화 획득
2. **분류**: 오디오 로드 → VGGish 특징 추출 → 분류기 모델 적용
3. **임베딩**: 오디오 로드 → 중간 모델 레이어에서 추출 → 다운스트림 작업 사용

## 오디오 처리

- 표준 샘플 레이트: 대부분 모델에서 16kHz
- 오디오 로딩: 적절한 `sampleRate` 및 `resampleQuality` 매개변수를 가진 `MonoLoader` 사용
- 모델은 특정 입력 형식 기대 (VGGish/MusiCNN용 mel-spectrograms)
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Essentia audio analysis project focused on music auto-tagging, classification, and embedding extraction using TensorFlow models. The project uses Essentia's TensorFlow integration to analyze audio features and make predictions about musical characteristics.

## Dependencies and Environment

- **Python**: ≥3.8 required
- **Package manager**: `uv` (uv.lock present)
- **Key dependencies**: 
  - `essentia>=2.1b6.dev1177` and `essentia-tensorflow>=2.1b6.dev1177` (core audio analysis)
  - `matplotlib>=3.7.5` (visualization)
  - `pandas>=2.0.3` (data handling)
  - `polars[pyarrow]>=1.8.2` (high-performance data processing)
  - `rich>=14.1.0` (CLI output)
  - `ffmpeg-python>=0.2.0` (audio processing)

## Project Structure

- `models/`: Pre-trained TensorFlow models (22 JSON files) for various audio analysis tasks
  - Mood classification: `mood_happy`, `mood_sad`, `mood_relaxed`, `mood_party`, etc.
  - Audio characteristics: `danceability`, `voice_instrumental`, `gender`, `timbre`
  - Music analysis: `msd-musicnn-1`, `moods_mirex`, `mtg_jamendo_moodtheme`
- `main.py`: Main execution script with audio processing pipeline
- `classifiers.py`: AudioClassifier class with 25+ prediction methods
- `utils.py`: Audio file processing utilities (loading, cutting)
- `feature_extractors.py`: FeatureExtractor class for audio feature extraction

## Core Architecture

**AudioClassifier**: Central classification class with specialized prediction methods:
- Model management: automatic download and metadata handling
- 25+ prediction methods covering emotions, music characteristics, and audio features
- Common prediction pipeline: `_common_predict()` method for consistent processing

**Key Essentia Components**:
- `TensorflowPredictMusiCNN`: For MusiCNN-based models (auto-tagging)
- `TensorflowPredictVGGish`: For VGGish-based models (classification, embeddings) 
- `TensorflowPredict2D`: For 2D prediction models
- `MonoLoader`: Audio loading with configurable sample rate (typically 16kHz)

## Running Code

- **Main script**: `uv run main.py` - processes audio files using AudioClassifier
- **Direct execution**: `python main.py` (within virtual environment)
- **Setup**: `uv sync` to install dependencies

## Model Usage Patterns

1. **Auto-tagging**: Load audio → Extract features with MusiCNN → Get tag activations
2. **Classification**: Load audio → Extract VGGish features → Apply classifier model
3. **Embeddings**: Load audio → Extract from intermediate model layers → Use for downstream tasks

## Audio Processing

- Standard sample rate: 16kHz for most models
- Audio loading: Use `MonoLoader` with appropriate `sampleRate` and `resampleQuality` parameters
- Models expect specific input formats (mel-spectrograms for VGGish/MusiCNN)

## Development Commands

```bash
# Environment setup
uv sync                    # Install dependencies
uv run main.py            # Run main script
python main.py            # Direct execution (in venv)
```

## Code Style
- **Classes**: PascalCase (AudioClassifier, FeatureExtractor)
- **Functions/Variables**: snake_case (predict_danceability, audio_files)
- **Private methods**: Leading underscore (_common_predict, _select_embeddings)
- **Architecture**: Class-based design with modular utilities

## Rules
- always answer in Korean
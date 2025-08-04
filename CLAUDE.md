# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Essentia audio analysis project focused on music auto-tagging, classification, and embedding extraction using TensorFlow models. The project uses Essentia's TensorFlow integration to analyze audio features and make predictions about musical characteristics.

## Dependencies and Environment

- **Python**: >=3.8 required
- **Key dependencies**: 
  - `essentia>=2.1b6.dev1177` and `essentia-tensorflow>=2.1b6.dev1177` (core audio analysis)
  - `matplotlib>=3.7.5` (visualization)
  - `pandas>=2.0.3` (data handling)
- **Package manager**: Uses `uv` (uv.lock present)

## Project Structure

- `models/`: Pre-trained TensorFlow models for various audio analysis tasks
  - `msd-musicnn-1.pb/json`: MusiCNN auto-tagging model trained on Million Song Dataset
  - `danceability-vggish-audioset-1.pb/json`: VGGish-based danceability classifier
  - `mood_*-vggish-*.pb`: Mood classification models (happy/sad)
  - `audioset-vggish-3.pb`: VGGish embedding extractor
- `audio/`: Audio files for testing/analysis (e.g., `techno_loop.wav`)
- `output/`: Generated output files

## Core Architecture

The project primarily uses Essentia's TensorFlow prediction algorithms:
- `TensorflowPredictMusiCNN`: For MusiCNN-based models (auto-tagging)
- `TensorflowPredictVGGish`: For VGGish-based models (classification, embeddings)
- `TensorflowPredict2D`: For 2D prediction models
- `MonoLoader`: Audio loading with configurable sample rate (typically 16kHz)

## Running Code

- **Main script**: `python test.py` - demonstrates mood classification using VGGish embeddings
- **Jupyter tutorial**: `tutorial_tensorflow_auto-tagging_classification_embeddings.ipynb` - comprehensive examples of auto-tagging, classification, and embedding extraction

## Model Usage Patterns

1. **Auto-tagging**: Load audio → Extract features with MusiCNN → Get tag activations
2. **Classification**: Load audio → Extract VGGish features → Apply classifier model
3. **Embeddings**: Load audio → Extract from intermediate model layers → Use for downstream tasks

## Audio Processing

- Standard sample rate: 16kHz for most models
- Audio loading: Use `MonoLoader` with appropriate `sampleRate` and `resampleQuality` parameters
- Models expect specific input formats (mel-spectrograms for VGGish/MusiCNN)

## Rules
- always answer in Korean
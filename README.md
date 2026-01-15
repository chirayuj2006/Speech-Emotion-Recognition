# Speech-Emotion-Recognition
A modular real-time Speech Emotion Recognition pipeline with completed audio capture, feature extraction, and preprocessing, ready for TensorFlow model integration.
The system captures live microphone audio, preprocesses it via a low-latency streaming pipeline, and extracts emotion-relevant acoustic features to generate model-ready inputs for a deep learning classifier. The architecture is proposed to be modular, extensible, and model-agnostic, allowing the easy integration of a trained TensorFlow/Keras model in the final stage.

Currently, the whole audio pipeline and preprocessing framework is ready, with only the training and integration of the neural network model to be completed.

## Project Development Approach

The project was developed in **two stages**:

1. **Baseline validation stage (`engine1.py`)**  
   A single-file implementation was created to verify the correctness of the complete real-time audio pipeline.

2. **Extension and model-integration stage (`engine.py`)**  
   After validating the pipeline, the system was extended toward cleaner orchestration and partial TensorFlow Lite integration.

This staged approach ensures correctness, stability, and extensibility.

---

## Baseline Implementation – `engine1.py`

`engine1.py` is a **self-contained reference implementation** that validates the complete real-time audio processing pipeline.

## Dependencies

- Python 3.x
- NumPy
- Librosa
- SoundDevice


### Functionality

- **Real-Time Audio Capture**
  - Live microphone input using `sounddevice`
  - Fixed sample rate (16 kHz) with block-based streaming

- **Buffering and Sliding Window Processing**
  - Continuous buffering of incoming audio blocks
  - Overlapping sliding windows for temporal context

- **Voice Activity Detection (VAD)**
  - RMS energy-based silence detection
  - Decibel thresholding to suppress non-speech segments

- **Feature Extraction**
  - MFCC features
  - Chroma features
  - Mel-spectrogram features
  - Time–frequency representations suitable for SER

- **Feature Stacking and Normalization**
  - Features stacked along the frequency axis
  - Global mean and standard deviation applied (computed offline)

- **Model Compatibility Validation**
  - TensorFlow Lite inference block included only to validate
    input shape compatibility
  - Model logic is not considered final at this stage

This file serves as a **stable baseline** and proof-of-correctness for the entire preprocessing pipeline.

---

## Extended Engine – `engine.py`

`engine.py` builds upon the validated baseline and moves the system closer to final deployment.

### Key Additions

- TensorFlow Lite interpreter initialization
- Explicit model input tensor shaping
- Inference pipeline scaffolding for emotion prediction

The deep learning model is **not finalized**, and emotion mapping is currently a placeholder.  
Model training, tuning, and final inference logic will be completed in the next phase.

---

## Dataset-Level Normalization

- Global mean and standard deviation are computed offline using NumPy
- Statistics are derived from MFCC, Chroma, and Mel features
- Stored as NumPy arrays and reused during inference
- Ensures consistency between training and real-time inference


---

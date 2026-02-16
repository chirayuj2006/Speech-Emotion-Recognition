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


## Code Structure and Functional Breakdown (`engine1.py`)

The file `engine1.py` implements the complete real-time audio preprocessing pipeline in a single, self-contained script.  
For clarity and verification, the code is organized into distinct functional sections, each responsible for a specific stage of the pipeline.  
This section documents those components individually and explains how they interact.

---

### Audio Stream Configuration

All real-time audio parameters are defined at the top of the file to control latency, resolution, and buffer sizes.  
These values determine how frequently audio is captured and how much temporal context is available for feature extraction.

```python
SAMPLE_RATE = 16000
CHANNELS = 1

BLOCK_DURATION = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

WINDOW_DURATION = 1.0
HOP_DURATION = 0.5

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)
```
### Audio Queue and Rolling Buffer Initialization

A thread-safe queue is used to decouple the real-time audio callback from the main processing loop. A rolling NumPy buffer accumulates incoming samples and enables sliding-window analysis.

```python
audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)
```


This design prevents audio dropouts and avoids performing expensive computation inside the callback thread.

### Real-Time Audio Callback

Audio capture is performed using a callback-based stream. The callback function is intentionally minimal and only pushes incoming audio blocks into the queue.

```python
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())
```


All downstream processing is deferred to the main loop.

### Voice Activity Detection (VAD)

Before feature extraction, each window is checked for speech activity using a simple RMS energy–based VAD. This suppresses silence and background noise and reduces unnecessary computation.
```python
def is_speech(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    db = 20 * np.log10(rms + EPSILON)
    return db > VAD_DB_THRESHOLD
```

The VAD logic is intentionally simple and deterministic, making it easy to replace with a more advanced method later.

### Feature Extraction Function

All acoustic feature computation is encapsulated inside a single function. This keeps the pipeline modular and makes future refactoring straightforward.

```python
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return mfcc, chroma, mel
```

MFCCs, chroma features, and mel-spectrograms are computed using consistent FFT and hop parameters to maintain temporal alignment

### Feature Alignment and Stacking

Extracted features are transposed so that time is the leading dimension. They are then stacked along the feature axis to form a unified representation.

```python
mfcc = mfcc.T
chroma = chroma.T
mel = mel.T

features = np.concatenate([mfcc, chroma, mel], axis=1)
```

This stacked tensor is suitable for direct input into convolutional or recurrent neural networks.

### Dataset-Level Normalization

Global mean and standard deviation values, computed offline, are applied to each feature window. This ensures consistency between training data and real-time inference.

```python
features = (features - mean) / (std + EPSILON)
```
Normalization statistics are loaded once and reused throughout execution.

### Main Processing Loop

The main loop coordinates audio ingestion, buffering, window extraction, VAD, feature computation, normalization, and optional model invocation.

```python
while True:
    block = audio_queue.get()
    audio_buffer = np.concatenate((audio_buffer, block))

    if len(audio_buffer) >= WINDOW_SIZE:
        window = audio_buffer[:WINDOW_SIZE]
        audio_buffer = audio_buffer[HOP_SIZE:]
```

The loop is designed for continuous execution with predictable latency and stable memory usage.

## engine.py — Additions and Extensions over engine1.py

The file `engine.py` extends the validated preprocessing pipeline implemented in `engine1.py`.  
All real-time audio capture, buffering, VAD, feature extraction, stacking, and normalization logic remain unchanged.

The purpose of `engine.py` is to move the system closer to deployment by introducing clearer orchestration and explicit model inference handling, while still keeping the model logic intentionally incomplete.

---

### 1. Explicit TensorFlow Lite Interpreter Initialization

Unlike `engine1.py`, where TensorFlow Lite is used only for shape validation, `engine.py` initializes the interpreter as a first-class component of the pipeline.

```python
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

### Dedicated Model Input Preparation

Feature tensors produced by the preprocessing pipeline are explicitly reshaped to match the expected model input format.
```python
model_input = features[np.newaxis, ..., np.newaxis].astype(np.float32)
```

This step enforces:

   1)batch dimension

   2)channel dimension (if required)

   3)correct data type for TensorFlow Lite execution

By making input shaping explicit, the inference stage becomes easier to debug and modify independently of feature extraction.
### Structured Inference Invocation

Inference execution is encapsulated as a clear, isolated step in the processing flow.
```python
interpreter.set_tensor(input_index, model_input)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
```

This separation ensures that model execution does not interfere with real-time audio handling and allows future extensions such as:
prediction smoothing,
temporal aggregation and 
confidence estimation

## Dataset-Level Normalization

Normalization statistics are computed offline using NumPy on the training dataset.
These statistics consist of a global mean and standard deviation calculated over MFCC, chroma, and mel-spectrogram features.

The computed values are stored as NumPy arrays and reused during real-time inference.
Applying the same normalization at inference time ensures consistency between training and deployment, preventing distribution mismatches that could affect model behavior.


---

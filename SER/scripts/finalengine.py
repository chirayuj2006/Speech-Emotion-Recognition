import sounddevice as sd
import numpy as np
import queue
import time
import librosa
import json
from tensorflow import keras


# ==============================
# AUDIO SETTINGS
# ==============================

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

WINDOW_DURATION = 1.0
HOP_DURATION = 0.5

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

N_FFT = 2048
HOP_LENGTH = 512

# MUST MATCH TRAINING
N_MFCC = 13
N_MELS = 11   # 13 + 12 + 11 = 36 total features

VAD_DB_THRESHOLD = -50
EPSILON = 1e-10

EMOTIONS = ["angry", "happy", "neutral", "sad"]  # Confirm order!


# ==============================
# LOAD MODEL (Keras)
# ==============================

with open("config.json", "r") as f:
    config = json.load(f)

model = keras.models.model_from_json(json.dumps(config))
model.load_weights("model.weights.h5")

print("Model loaded successfully")
print("Expected input shape:", model.input_shape)


# ==============================
# NORMALIZATION (ONLY IF TRAINED WITH IT)
# ==============================

try:
    FEATURE_MEAN = np.load("feature_mean.npy")
    FEATURE_STD = np.load("feature_std.npy")
    NORMALIZE = True
    print("Normalization enabled")
except:
    NORMALIZE = False
    print("No normalization file found. Using raw features.")


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC
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
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # TAKE MEAN ACROSS TIME
    mfcc_mean = np.mean(mfcc, axis=1)     # (13,)
    chroma_mean = np.mean(chroma, axis=1) # (12,)
    mel_mean = np.mean(mel, axis=1)       # (11,)

    features = np.concatenate([mfcc_mean, chroma_mean, mel_mean])

    return features  # (36,)


# ==============================
# REAL-TIME AUDIO ENGINE
# ==============================

audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().flatten())


stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)

print("Microphone started. Speak.")
stream.start()


try:
    while True:
        if not audio_queue.empty():
            block = audio_queue.get()
            audio_buffer = np.concatenate((audio_buffer, block))

            if len(audio_buffer) >= WINDOW_SIZE:

                window = audio_buffer[:WINDOW_SIZE]

                # VAD
                rms = np.sqrt(np.mean(window**2))
                db = 20 * np.log10(rms + EPSILON)

                if db < VAD_DB_THRESHOLD:
                    print("Silence")
                else:
                    features = extract_features(window, SAMPLE_RATE)

                    # NORMALIZATION
                    if NORMALIZE:
                        features = (features - FEATURE_MEAN) / (FEATURE_STD + 1e-8)

                    # Reshape to (1,36)
                    model_input = features.reshape(1, 36)

                    prediction = model.predict(model_input, verbose=0)[0]

                    emotion_idx = np.argmax(prediction)
                    confidence = np.max(prediction)

                    print(f"Emotion: {EMOTIONS[emotion_idx]} | Confidence: {confidence:.2f}")

                audio_buffer = audio_buffer[HOP_SIZE:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop()
    stream.close()

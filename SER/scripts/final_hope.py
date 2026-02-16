

import sounddevice as sd
import numpy as np
import queue
import time
import librosa
import tensorflow as tf



SAMPLE_RATE = 16000
CHANNELS = 1

BLOCK_DURATION = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

WINDOW_DURATION = 3.0
HOP_DURATION = 1.5

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

VAD_DB_THRESHOLD = -40
EPSILON = 1e-10



N_FFT = 2048
HOP_LENGTH = 256

N_MFCC = 48
N_MELS = 48

TARGET_TIME_FRAMES = 180
TOTAL_FEATURES = 108


FEATURE_MEAN = np.load("mean_yay.npy")   # (108,)
FEATURE_STD = np.load("std_yay.npy")     # (108,)



def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(108, 180)),

        tf.keras.layers.Conv1D(32, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.LSTM(64),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax")
    ])

    return model


model = build_model()

model.load_weights("model.weights.h5")

print(" Model loaded successfully")


EMOTIONS = ["angry", "happy", "neutral", "sad"]


audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)



def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC
    )

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel)

    return mfcc, chroma, mel


def stack_features(mfcc, chroma, mel):
    return np.vstack([mfcc, chroma, mel])  # -> (108, T)


def fix_time_dimension(features):
    """Pad or trim to match model time frames (180)."""

    T = features.shape[1]

    if T < TARGET_TIME_FRAMES:
        pad_width = TARGET_TIME_FRAMES - T
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

    elif T > TARGET_TIME_FRAMES:
        features = features[:, :TARGET_TIME_FRAMES]

    return features


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

print(" Microphone started... Speak now.")
stream.start()


try:
    while True:
        if not audio_queue.empty():
            block = audio_queue.get()
            audio_buffer = np.concatenate((audio_buffer, block))

            if len(audio_buffer) >= WINDOW_SIZE:
                window = audio_buffer[:WINDOW_SIZE]

              
                rms = np.sqrt(np.mean(window ** 2))
                db = 20 * np.log10(rms + EPSILON)

                if db < VAD_DB_THRESHOLD:
                    print("Silence detected")

                else:
                    mfcc, chroma, mel = extract_features(window, SAMPLE_RATE)
                    features = stack_features(mfcc, chroma, mel)

               
                    features = fix_time_dimension(features)

                
                    features_norm = (
                        features - FEATURE_MEAN[:, None]
                    ) / (FEATURE_STD[:, None] + 1e-8)


                    model_input = np.expand_dims(features_norm, axis=0)


                    prediction = model.predict(model_input, verbose=0)[0]

                    emotion_idx = np.argmax(prediction)
                    emotion = EMOTIONS[emotion_idx]

                    print("Emotion:", emotion)

                
                audio_buffer = audio_buffer[HOP_SIZE:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop()
    stream.close()

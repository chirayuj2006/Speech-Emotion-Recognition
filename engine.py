import sounddevice as sd
import numpy as np
import queue
import time
import numpy as np
import librosa
import tensorflow as tf



SAMPLE_RATE = 16000      
CHANNELS = 1
BLOCK_DURATION = 0.1     # 100 ms blocks
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
WINDOW_DURATION = 1.0      
HOP_DURATION = 0.5    

N_FFT = 2048
HOP_LENGTH = 512

N_MFCC = 128     
N_MELS = 15      



WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)
VAD_DB_THRESHOLD = -40  
EPSILON = 1e-10         

def extract_features(audio, sr):
    # (128, T )
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC
    )

    # (12, T)
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    #   (15,  T)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    return mfcc, chroma, mel

def stack_features(mfcc, chroma, mel):
    """
    All inputs: (features, T)
    Output: (155, T)
    """
    return np.vstack([mfcc, chroma, mel])

FEATURE_MEAN = np.load("mean.npy")   # ( 155,)
FEATURE_STD = np.load("std.npy")     # (155,)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model input shape:", input_details[0]["shape"])


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

print("Microphone started. Speak")

stream.start()

try:
    while True:
        if not audio_queue.empty():
            block = audio_queue.get()
            audio_buffer = np.concatenate((audio_buffer, block))

            if len(audio_buffer) >= WINDOW_SIZE:
                window = audio_buffer[:WINDOW_SIZE]

                # ===== VAD =====
                rms = np.sqrt(np.mean(window**2))
                db = 20 * np.log10(rms + EPSILON)

                if db < VAD_DB_THRESHOLD:
                    print("Silence")
                else:
                    # ===== FEATURE EXTRACTION =====
                    mfcc, chroma, mel = extract_features(window, SAMPLE_RATE)

                    # ===== STACK FEATURES =====
                    features = stack_features(mfcc, chroma, mel)  # (155, T)

                    # ===== NORMALIZATION =====
                    features_norm = (
                        features - FEATURE_MEAN[:, None]
                    ) / (FEATURE_STD[:, None] + 1e-8)

                    # ===== MODEL INPUT SHAPING =====
                    model_input = features_norm[np.newaxis, :, :, np.newaxis]

                    # ===== INFERENCE =====
                    interpreter.set_tensor(
                        input_details[0]["index"],
                        model_input.astype(np.float32)
                    )
                    interpreter.invoke()

                    prediction = interpreter.get_tensor(
                        output_details[0]["index"]
                    )[0]

                    emotion_idx = np.argmax(prediction)
                    emotion = EMOTIONS[emotion_idx]

                    print("Emotion:", emotion)

                # ===== SLIDE WINDOW =====
                audio_buffer = audio_buffer[HOP_SIZE:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop()
    stream.close()



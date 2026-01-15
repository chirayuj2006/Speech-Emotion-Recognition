import numpy as np
import librosa

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 128
N_MELS = 15

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

    return mfcc, chroma, mel

def stack_features(mfcc, chroma, mel):
    return np.vstack([mfcc, chroma, mel])

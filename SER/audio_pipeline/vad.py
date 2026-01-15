import numpy as np

EPSILON = 1e-10

def is_speech(audio_window, threshold_db):
    rms = np.sqrt(np.mean(audio_window ** 2))
    db = 20 * np.log10(rms + EPSILON)
    return db >= threshold_db, db

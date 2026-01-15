import numpy as np

def update_buffer(audio_buffer, new_block):
    return np.concatenate((audio_buffer, new_block))

def has_full_window(audio_buffer, window_size):
    return len(audio_buffer) >= window_size

def get_window(audio_buffer, window_size):
    return audio_buffer[:window_size]

def slide_buffer(audio_buffer, hop_size):
    return audio_buffer[hop_size:]

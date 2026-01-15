import numpy as np

def load_stats(mean_path, std_path):
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std

def normalize(features, mean, std):
    return (features - mean[:, None]) / (std[:, None] + 1e-8)

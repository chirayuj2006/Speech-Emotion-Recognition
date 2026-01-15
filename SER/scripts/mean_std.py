import pandas as pd
import numpy as np
import ast

df = pd.read_csv("labels.csv")

# training set only
train_df = df.iloc[:7000]

all_features = []

for _, row in train_df.iterrows():
    # Convert string → list → numpy array
    mfcc = np.array(ast.literal_eval(row["mfccs"]))
    chroma = np.array(ast.literal_eval(row["chroma"]))
    mel = np.array(ast.literal_eval(row["mel"]))

    # Stack features (180, T)
    features = np.vstack([mfcc, chroma, mel])
    all_features.append(features)

# Concatenate across time
all_features = np.concatenate(all_features, axis=1)

# Compute mean and std
mean = np.mean(all_features, axis=1)
std = np.std(all_features, axis=1)

np.save("mean.npy", mean)
np.save("std.npy", std)

print("mean shape:", mean.shape)
print("std shape:", std.shape)

import numpy as np

X = np.load("labels.npy", allow_pickle=True)

all_features = []

for sample in X:

    mfcc = sample[2]     
    chroma = sample[3]   
    mel = sample[4]      

    mfcc = mfcc.T        
    chroma = chroma.T    
    mel = mel.T          

    features = np.vstack([mfcc, chroma, mel])
    all_features.append(features)

all_features = np.concatenate(all_features, axis=1)  

mean = np.mean(all_features, axis=1)
std = np.std(all_features, axis=1)


np.save("mean.npy", mean)
np.save("std.npy", std)

print("mean shape:", mean.shape)
print("std shape:", std.shape)

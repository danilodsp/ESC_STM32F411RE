### extract_features.py

import os
import numpy as np
import librosa
import soundfile as sf

SR = 16000
N_MELS = 40
DURATION = 1.0  # seconds
def extract_logmel(audio_path, sr=SR, n_mels=N_MELS):
    y, _ = librosa.load(audio_path, sr=sr)
    y = librosa.util.fix_length(y, int(sr * DURATION))
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel).T

def process_directory(base_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(subdir, file)
                label = os.path.basename(subdir)
                feature = extract_logmel(path)
                np.save(os.path.join(out_dir, f"{label}_{file}.npy"), feature)
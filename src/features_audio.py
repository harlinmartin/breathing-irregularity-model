import os
import librosa
import numpy as np
from tqdm import tqdm

# -------------------
# Paths
# -------------------
AUDIO_DIR = r"E:\breathing\breathing_irregularity_model\data\audio\ICBHI_final_database"
PROCESSED_DIR = "../data/processed"

N_MFCC = 40
SR = 16000
MAX_LEN = 300  # number of time frames to pad/truncate sequences

# -------------------
# Utils
# -------------------
def pad_or_truncate(seq, max_len=MAX_LEN):
    """Ensure MFCC sequence has fixed length."""
    if seq.shape[1] < max_len:
        pad_width = max_len - seq.shape[1]
        return np.pad(seq, ((0,0),(0,pad_width)), mode="constant")
    else:
        return seq[:, :max_len]

def augment_audio(y, sr):
    """Return list of augmented signals (original + aug)."""
    signals = [y]  # original

    # Noise
    noise = np.random.normal(0, 0.005, y.shape)
    signals.append(y + noise)

    # Pitch shift
    try:
        signals.append(librosa.effects.pitch_shift(y, sr, n_steps=2))
        signals.append(librosa.effects.pitch_shift(y, sr, n_steps=-2))
    except:
        pass

    # Time stretch
    try:
        signals.append(librosa.effects.time_stretch(y, 0.9))
        signals.append(librosa.effects.time_stretch(y, 1.1))
    except:
        pass

    return signals

def extract_features(file_path, sr=SR, n_mfcc=N_MFCC):
    """Extract MFCC sequences for original + augmented versions."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)

        signals = augment_audio(y, sr)
        feats = []

        for sig in signals:
            mfccs = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)
            mfccs = pad_or_truncate(mfccs, MAX_LEN)
            feats.append(mfccs)

        return feats
    except Exception as e:
        print(f"⚠️ Could not process {file_path}: {e}")
        return []

def get_label_from_txt(txt_path):
    """Read annotation and assign label: 0 = Normal, 1 = Abnormal."""
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 4:
                crackle, wheeze = int(parts[2]), int(parts[3])
                if crackle == 1 or wheeze == 1:
                    return 1
        return 0
    except:
        return 0

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    features, labels = [], []

    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

    for file in tqdm(wav_files, desc="🔎 Extracting features", unit="file"):
        wav_path = os.path.join(AUDIO_DIR, file)
        txt_path = wav_path.replace(".wav", ".txt")

        feats_list = extract_features(wav_path)
        if feats_list:
            label = get_label_from_txt(txt_path)
            for feat in feats_list:
                features.append(feat)
                labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    print(f"\n✅ Final features shape: {features.shape} (samples, mfcc, time)")
    print(f"✅ Final labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels, return_counts=True)}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "features_seq.npy"), features)
    np.save(os.path.join(PROCESSED_DIR, "labels_seq.npy"), labels)

    print(f"\n📂 Saved sequential features (with augmentation) to {PROCESSED_DIR}")

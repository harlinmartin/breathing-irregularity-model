import numpy as np
import librosa
import tensorflow as tf
import scipy.signal as signal

# -------------------------
# Load Trained Model
# -------------------------
model = tf.keras.models.load_model("../models/best_model.keras")

# -------------------------
# Feature Extraction (same as training)
# -------------------------
N_MFCC = 40
SR = 16000
MAX_LEN = 300

def pad_or_truncate(seq, max_len=MAX_LEN):
    if seq.shape[1] < max_len:
        pad_width = max_len - seq.shape[1]
        return np.pad(seq, ((0,0),(0,pad_width)), mode="constant")
    else:
        return seq[:, :max_len]

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    y = librosa.util.normalize(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfccs = pad_or_truncate(mfccs, MAX_LEN)
    return mfccs

# -------------------------
# Breathing Rate Estimation
# -------------------------
def estimate_breathing_rate(file_path, sr=SR):
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)

    # Bandpass filter (100–1000 Hz typical for breath sounds)
    sos = signal.butter(4, [100, 1000], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, y)

    # Envelope extraction
    envelope = np.abs(signal.hilbert(y))
    envelope = librosa.util.normalize(envelope)

    # Smooth
    envelope_smooth = signal.medfilt(envelope, kernel_size=201)

    # Peak detection (min ~0.8s apart)
    peaks, _ = signal.find_peaks(envelope_smooth, distance=sr*0.8)

    duration_sec = len(y) / sr
    num_breaths = len(peaks)
    breaths_per_min = (num_breaths / duration_sec) * 60
    return breaths_per_min, num_breaths, duration_sec

# -------------------------
# Run on Single Audio File
# -------------------------
if __name__ == "__main__":
    test_file = r"E:\Breathing irregularity\archive (1)\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files\104_1b1_Pl_sc_Litt3200.wav"  # 🔹 replace with your test file path

    # Classifier prediction
    feat = extract_features(test_file)
    feat = feat[np.newaxis, ..., np.newaxis]
    probs = model.predict(feat)
    pred_class = np.argmax(probs, axis=1)[0]
    confidence = np.max(probs)

    # Breathing rate estimation
    br, n_breaths, dur = estimate_breathing_rate(test_file)

    # Results
    print(f"\n✅ Predicted class: {pred_class} (0=Normal, 1=Abnormal)")
    print(f"✅ Confidence: {confidence:.4f}")
    print(f"✅ Estimated Breathing Rate: {br:.1f} breaths/min "
          f"({n_breaths} breaths over {dur:.1f} sec)")

# Breathing Irregularity Detection Model

This project uses deep learning and audio signal processing to classify **normal vs abnormal breathing sounds** and also estimate **breathing rate** from respiratory audio recordings.

---

---

## ⚙️ Workflow

1. **Feature Extraction (`features_audio.py`)**
   - Uses `librosa` to extract **MFCC (Mel-frequency cepstral coefficients)** features.
   - Applies data augmentation:
     - Gaussian noise
     - Pitch shift
     - Time stretch
     - SpecAugment-style masking
   - Labels are generated from ICBHI dataset annotations.

2. **Model Training (`train.py`)**
   - CNN-based model with Conv2D, BatchNorm, MaxPooling, and GlobalAveragePooling.
   - Data balancing using `RandomOverSampler`.
   - Training-time augmentation (noise, time-shift, scaling, masking).
   - Early stopping + learning rate scheduler + model checkpoint.
   - Metrics: Accuracy, Balanced Accuracy, Macro F1, Confusion Matrix.

3. **Inference (`test_single_audio.py`)**
   - Load a trained `.keras` model.
   - Extract MFCC from a single `.wav` file.
   - Predict class:
     - `0 = Normal`
     - `1 = Abnormal`
   - Outputs prediction with confidence.

---

## 🫁 Breathing Rate Calculation

The breathing rate is calculated separately from classification.

### 🔹 Formula:
Breathing Rate (BPM)=Duration (seconds)Number of Breaths​×60

### 🔹 Steps:
1. Preprocess audio → band-pass filter around typical breathing frequencies (0.1–2 Hz).
2. Compute envelope using **Hilbert transform** or **spectral energy**.
3. Detect **peaks** (each peak ≈ 1 inhalation/exhalation cycle).
4. Count number of cycles → convert to breaths per minute.

---

## 📊  Results

- **Classification Accuracy (Balanced):** ~0.89  
- **Macro F1 Score:** ~0.88  
- Confusion Matrix shows good balance between Normal vs Abnormal.  
- Training curves (accuracy & loss) are stable, no major overfitting.

---

## 🚀 How to Run

### Train the model:
```bash
python src/train.py
```
### Test a single audio file:
```bash
python src/test_single_audio.py --file path/to/sample.wav
```


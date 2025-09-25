import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt


# -------------------------
# Augmentation function
# -------------------------
def augment_data(X, y):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        # original
        X_aug.append(X[i])
        y_aug.append(y[i])

        # 1. Gaussian noise
        noise = np.random.normal(0, 0.01, X[i].shape)
        X_aug.append(X[i] + noise)
        y_aug.append(y[i])

        # 2. Time-shift
        shift = np.random.randint(1, X[i].shape[0] // 10)
        X_aug.append(np.roll(X[i], shift, axis=0))
        y_aug.append(y[i])

        # 3. Scaling
        scale = np.random.uniform(0.9, 1.1)
        X_aug.append(X[i] * scale)
        y_aug.append(y[i])

        # 4. SpecAugment-style masking
        mask = X[i].copy()
        t = np.random.randint(1, mask.shape[0] // 5)
        start = np.random.randint(0, mask.shape[0] - t)
        mask[start:start + t, :] = 0
        X_aug.append(mask)
        y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


# -------------------------
# Load data
# -------------------------
X = np.load(r"E:\breathing\breathing_irregularity_model\data\processed\features_seq.npy")
y = np.load(r"E:\breathing\breathing_irregularity_model\data\processed\labels_seq.npy")

print(f"✅ Features shape: {X.shape}")
print(f"✅ Labels shape: {y.shape}")

n_classes = len(np.unique(y))

# -------------------------
# Oversample (requires 2D)
# -------------------------
X_reshaped = X.reshape((X.shape[0], -1))  # flatten
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_reshaped, y)

# reshape back to original 3D
X = X_resampled.reshape((-1, X.shape[1], X.shape[2]))
y = y_resampled

print(f"✅ After oversampling: {X.shape}, {y.shape}, class balance: {np.bincount(y)}")

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply augmentation only on training data
X_train, y_train = augment_data(X_train, y_train)
print(f"✅ After augmentation: {X_train.shape}, {y_train.shape}")

# Add channel dim for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# -------------------------
# Model
# -------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(X.shape[1], X.shape[2], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Callbacks
# -------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint("../models/best_model.keras", save_best_only=True)
]

# -------------------------
# Train
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# -------------------------
# Evaluate
# -------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("✅ Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("✅ Macro F1:", f1_score(y_test, y_pred, average="macro"))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Training Curves
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Training Curves")
plt.show()

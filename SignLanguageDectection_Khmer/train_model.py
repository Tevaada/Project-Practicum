"""
train_model.py — Train the LSTM sign language model.

Data split : 70% train | 15% validation | 15% test
Optimizer  : AdamW  lr=0.001
Epochs     : 300
Batch size : 8
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from utils import DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, MODEL_PATH, build_model

# ── Load data ──────────────────────────────────────────────────────────────────
label_map = {label: i for i, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    for seq in range(NO_SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            path = os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                print("Run collect_data.py first.")
                exit()
            window.append(np.load(path))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
print(f"Dataset  X:{X.shape}  y:{y.shape}")

# ── 70 / 15 / 15 split ────────────────────────────────────────────────────────
# Step 1: split off 30% as temp (will become val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Step 2: split the 30% temp equally into 15% val and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────────────────────────────────
model = build_model()
model.summary()

model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=8,
    validation_data=(X_val, y_val),
    callbacks=[
        TensorBoard(log_dir="Logs"),
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
    ],
    verbose=1,
)

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ── Evaluate on held-out test set ─────────────────────────────────────────────
yhat  = model.predict(X_test, verbose=0)
ytrue = np.argmax(y_test,  axis=1)
ypred = np.argmax(yhat,    axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(ytrue, ypred))
print(f"Test Accuracy: {accuracy_score(ytrue, ypred):.4f}")

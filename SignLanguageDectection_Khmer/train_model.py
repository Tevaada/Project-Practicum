"""
train_model.py — Train the LSTM sign language model.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import TensorBoard
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────────────────────────────────
model = build_model()
model.summary()

model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=8,
    callbacks=[TensorBoard(log_dir="Logs")],
    verbose=1,
)

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ── Evaluate ───────────────────────────────────────────────────────────────────
yhat  = model.predict(X_test, verbose=0)
ytrue = np.argmax(y_test, axis=1)
ypred = np.argmax(yhat,   axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(ytrue, ypred))
print(f"Accuracy: {accuracy_score(ytrue, ypred):.4f}")

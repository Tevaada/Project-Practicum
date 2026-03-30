"""
train_model.py — Train the LSTM sign language model.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from utilities import DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, MODEL_PATH, LOG_PATH, build_model

# ── Load data ──────────────────────────────────────────────────────────────────
label_map = {label: i for i, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    action_sequences = []
    skip_action = False

    for seq in range(NO_SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            path = os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy")
            if not os.path.exists(path):
                print(f"  [SKIP] Missing: {path}")
                skip_action = True
                break
            window.append(np.load(path))
        if skip_action:
            break
        action_sequences.append(window)

    if skip_action:
        print(f"  [SKIP] Action '{action}' incomplete — run collect_data.py first.")
        continue

    sequences.extend(action_sequences)
    labels.extend([label_map[action]] * len(action_sequences))
    print(f"  [OK] '{action}' — {len(action_sequences)} sequences loaded")

if len(sequences) == 0:
    print("\nNo data loaded. Run collect_data.py first.")
    exit()

X = np.array(sequences, dtype=np.float32)
y_int = np.array(labels)
y = to_categorical(y_int, num_classes=len(ACTIONS)).astype(np.float32)
print(f"\nDataset  X:{X.shape}  y:{y.shape}")

# ── Split first, then augment only training data ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Augment training data with multiple noise levels ──────────────────────────
def augment(X, y):
    parts_X, parts_y = [X], [y]
    for std in [0.003, 0.006, 0.010]:
        noise = np.random.normal(0, std, X.shape).astype(np.float32)
        parts_X.append(X + noise)
        parts_y.append(y)
    return np.concatenate(parts_X), np.concatenate(parts_y)

X_train, y_train = augment(X_train, y_train)

# Shuffle
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

print(f"Train (augmented): {len(X_train)}  Test: {len(X_test)}")

# ── Class weights to prevent bias ─────────────────────────────────────────────
y_train_int = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_int),
    y=y_train_int,
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ── Callbacks ──────────────────────────────────────────────────────────────────
callbacks = [
    TensorBoard(log_dir=LOG_PATH),
    EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=50,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_categorical_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# ── Train ──────────────────────────────────────────────────────────────────────
model = build_model()
model.summary()

model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)

print(f"\nBest model saved → {MODEL_PATH}")

# ── Evaluate ───────────────────────────────────────────────────────────────────
yhat  = model.predict(X_test, verbose=0)
ytrue = np.argmax(y_test, axis=1)
ypred = np.argmax(yhat,   axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(ytrue, ypred))
print(f"\nAccuracy: {accuracy_score(ytrue, ypred):.4f}")
print("\nPer-class Report:")
print(classification_report(ytrue, ypred, target_names=ACTIONS))

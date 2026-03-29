import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# ── Settings ───────────────────────────────────────────────────────────────────
DATA_PATH       = "MP_Data"
ACTIONS         = np.array(["hello", "thanks"])
NO_SEQUENCES    = 30          # more data = better model
SEQUENCE_LENGTH = 20
MODEL_PATH      = "action_model.keras"
THRESHOLD       = 0.60        # lowered — model needs room to breathe
SMOOTH_FRAMES   = 5

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def extract_keypoints(results):
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility]
                  for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z]
                  for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def create_folders():
    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)


def build_model():
    """LSTM model — simplified for small datasets."""
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1662)),
        LSTM(64, return_sequences=True, activation="relu"),
        LSTM(128, return_sequences=False, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(len(ACTIONS), activation="softmax"),
    ])
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


class PredictionSmoother:
    """Average the last N probability vectors to reduce prediction flickering."""

    def __init__(self, size=SMOOTH_FRAMES):
        self._buf = deque(maxlen=size)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self._buf.append(probs)
        return np.mean(self._buf, axis=0)

    def reset(self):
        self._buf.clear()

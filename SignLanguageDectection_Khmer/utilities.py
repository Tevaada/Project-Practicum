import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Base directory (always relative to this file)
_BASE = os.path.dirname(os.path.abspath(__file__))

# Settings
DATA_PATH       = os.path.join(_BASE, "Dataset", "MediaPipe_Data")
AUDIO_PATH      = os.path.join(_BASE, "Dataset", "Audio")
ACTIONS         = np.array(["hello", "bye bye", "i love you", "nothing"])
NO_SEQUENCES    = 30
SEQUENCE_LENGTH = 20
MODEL_PATH      = os.path.join(_BASE, "action_model.keras")
LOG_PATH        = os.path.join(_BASE, "Logs")
THRESHOLD       = 0.60
SMOOTH_FRAMES   = 5

KHMER_SPEECH = {
    "hello":      "សួស្តី",
    "bye bye":    "លាហើយ",
    "i love you": "ខ្ញុំស្រឡាញ់អ្នក",
}

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
    reg = l2(1e-4)
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1662)),
        LSTM(64, return_sequences=True, activation="tanh",
             kernel_regularizer=reg, recurrent_regularizer=reg),
        Dropout(0.4),
        LSTM(128, return_sequences=False, activation="tanh",
             kernel_regularizer=reg, recurrent_regularizer=reg),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu", kernel_regularizer=reg),
        Dropout(0.3),
        Dense(len(ACTIONS), activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


class PredictionSmoother:
    def __init__(self, size=SMOOTH_FRAMES):
        self._buf = deque(maxlen=size)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self._buf.append(probs)
        return np.mean(self._buf, axis=0)

    def reset(self):
        self._buf.clear()
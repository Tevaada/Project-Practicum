"""
predict.py — Real-time sign language detection with custom audio support.

How it works:
  - Signs are tracked continuously in a rolling buffer.
  - The current best sign is shown live but NOT spoken yet.
  - Press SPACE (or let the sign drop to "No sign") to CONFIRM the sign,
    add it to the sentence, and play its audio.

Controls:
  Q / ESC — quit
  C       — clear sentence
  S       — toggle speech on/off
  SPACE   — confirm current sign into sentence and speak it

Audio:
  Place audio files in the  audio/  folder named after each action.
  Supported formats: .mp3  .wav  .ogg
  Example:  audio/hello.mp3   audio/thanks.wav
  If no file is found, falls back to gTTS online TTS.
"""
import os
import cv2
import tempfile
import threading
import numpy as np
import pygame
from tensorflow.keras.models import load_model

from utils import (
    ACTIONS, MODEL_PATH, SEQUENCE_LENGTH, THRESHOLD, KHMER_SPEECH,
    mp_holistic, mediapipe_detection, draw_styled_landmarks,
    extract_keypoints, PredictionSmoother,
)

# ── Audio setup ────────────────────────────────────────────────────────────────
AUDIO_DIR      = "audio"
LANG           = "km"
speech_enabled = [True]

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
_speech_lock = threading.Lock()


def _find_audio_file(action: str):
    """Return path to a local audio file for this action, or None."""
    for ext in (".mp3", ".wav", ".ogg"):
        path = os.path.join(AUDIO_DIR, action + ext)
        if os.path.exists(path):
            return path
    return None


def speak(action: str):
    """Play local audio if available, otherwise fall back to gTTS."""
    if not speech_enabled[0]:
        return

    local = _find_audio_file(action)
    text  = KHMER_SPEECH.get(action, action)

    def _run():
        with _speech_lock:
            try:
                if local:
                    sound = pygame.mixer.Sound(local)
                    sound.play()
                    pygame.time.wait(int(sound.get_length() * 1000) + 100)
                else:
                    # fallback: gTTS
                    from gtts import gTTS
                    tts = gTTS(text=text, lang=LANG)
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tmp = f.name
                    tts.save(tmp)
                    sound = pygame.mixer.Sound(tmp)
                    sound.play()
                    pygame.time.wait(int(sound.get_length() * 1000) + 100)
                    os.remove(tmp)
            except Exception as e:
                print(f"\nSpeech error: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ── UI ─────────────────────────────────────────────────────────────────────────
PANEL_W = 300
ACCENT  = (0, 255, 255)
GREEN   = (0, 220, 0)
ORANGE  = (0, 165, 255)
RED     = (0, 0, 220)
WHITE   = (255, 255, 255)
GREY    = (150, 150, 150)
DARK    = (28, 28, 28)


def _bar(canvas, x, y, w, h, value, color):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (65, 65, 65), -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * min(float(value), 1.0)), y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_panel(frame, word, conf, sentence, probs, pending: bool):
    h, fw = frame.shape[:2]
    canvas = np.full((h, fw + PANEL_W, 3), DARK, dtype=np.uint8)
    canvas[:, :fw] = frame

    px    = fw + 12
    bar_w = PANEL_W - 24

    # title
    cv2.putText(canvas, "SIGN DETECTION", (px, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ACCENT, 2, cv2.LINE_AA)
    cv2.line(canvas, (px, 42), (fw + PANEL_W - 12, 42), (55, 55, 55), 1)

    # detected word — blink yellow when pending confirmation
    word_color = (0, 200, 255) if pending else (GREEN if word != "No sign" else GREY)
    cv2.putText(canvas, word, (px, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, word_color, 2, cv2.LINE_AA)
    if pending:
        cv2.putText(canvas, "SPACE to confirm", (px, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

    # confidence bar
    cv2.putText(canvas, f"Confidence  {conf:.0%}", (px, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, GREY, 1, cv2.LINE_AA)
    _bar(canvas, px, 128, bar_w, 12, conf, GREEN if conf >= THRESHOLD else ORANGE)
    tx = px + int(bar_w * THRESHOLD)
    cv2.line(canvas, (tx, 126), (tx, 142), RED, 2)
    cv2.putText(canvas, f"{THRESHOLD:.0%}", (tx - 14, 154),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, RED, 1, cv2.LINE_AA)

    # per-action probability bars
    cv2.putText(canvas, "Probabilities", (px, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, GREY, 1, cv2.LINE_AA)
    for i, (act, p) in enumerate(zip(ACTIONS, probs)):
        by = 179 + i * 30
        _bar(canvas, px, by, bar_w, 18, float(p), (90, 160, 255))
        # show speaker icon if local audio exists
        audio_tag = " ♪" if _find_audio_file(act) else ""
        cv2.putText(canvas, f"{act}{audio_tag}  {p:.0%}", (px + 4, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1, cv2.LINE_AA)

    # sentence box
    sep_y = 179 + len(ACTIONS) * 30 + 14
    cv2.line(canvas, (px, sep_y), (fw + PANEL_W - 12, sep_y), (55, 55, 55), 1)
    cv2.putText(canvas, "Sentence", (px, sep_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, GREY, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (px, sep_y + 24), (fw + PANEL_W - 12, sep_y + 58), (48, 48, 48), -1)
    cv2.putText(canvas, " ".join(sentence[-5:]) or "...", (px + 6, sep_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREEN, 2, cv2.LINE_AA)

    # bottom controls
    bot = h - 115
    cv2.line(canvas, (px, bot), (fw + PANEL_W - 12, bot), (55, 55, 55), 1)
    cv2.putText(canvas, f"Speech: {'ON' if speech_enabled[0] else 'OFF'}",
                (px, bot + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                GREEN if speech_enabled[0] else RED, 2, cv2.LINE_AA)
    for i, hint in enumerate(["SPACE  confirm sign", "C  clear", "S  speech on/off", "Q/ESC  quit"]):
        cv2.putText(canvas, hint, (px, bot + 44 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)

    return canvas


# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"Model not found: '{MODEL_PATH}' — run train_model.py first.")
    exit()

model = load_model(MODEL_PATH)
print(f"Model loaded  |  actions: {list(ACTIONS)}  |  threshold: {THRESHOLD}")

# Show which actions have local audio
for act in ACTIONS:
    f = _find_audio_file(act)
    print(f"  [{act}]  audio: {f if f else 'none (will use gTTS)'}")

# ── State ──────────────────────────────────────────────────────────────────────
sequence      = []
sentence      = []
display_word  = "No sign"
confidence    = 0.0
last_probs    = np.zeros(len(ACTIONS))
smoother      = PredictionSmoother()

# Tracks the last confirmed sign so we don't auto-confirm the same sign twice
last_confirmed = None

# How many consecutive "No sign" frames before we auto-confirm a pending sign
NO_SIGN_CONFIRM_FRAMES = 15
no_sign_counter        = 0

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

print("\nRunning — perform a sign, then press SPACE (or drop hands) to confirm it\n")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        sequence.append(extract_keypoints(results))
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            raw        = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            smoothed   = smoother.update(raw)
            idx        = int(np.argmax(smoothed))
            confidence = float(smoothed[idx])
            predicted  = ACTIONS[idx]
            last_probs = smoothed

            probs_str = "  ".join(f"{a}:{p:.2f}" for a, p in zip(ACTIONS, smoothed))
            print(f"\r{probs_str}  → {predicted} ({confidence:.2f})", end="", flush=True)

            if confidence >= THRESHOLD:
                display_word    = predicted
                no_sign_counter = 0
            else:
                # Sign dropped — auto-confirm after enough "no sign" frames
                if display_word != "No sign":
                    no_sign_counter += 1
                    if no_sign_counter >= NO_SIGN_CONFIRM_FRAMES:
                        # auto-confirm
                        if display_word != last_confirmed:
                            sentence.append(display_word)
                            speak(display_word)
                            print(f"\nAuto-confirmed: {display_word}")
                            last_confirmed = display_word
                        display_word    = "No sign"
                        no_sign_counter = 0
                        smoother.reset()
                else:
                    no_sign_counter = 0
                    display_word    = "No sign"
                    smoother.reset()

        # pending = a sign is showing but not yet confirmed
        pending = display_word != "No sign"

        cv2.imshow("Sign Detection",
                   draw_panel(image, display_word, confidence, sentence, last_probs, pending))

        key = cv2.waitKey(10) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            if display_word != "No sign":
                sentence.append(display_word)
                speak(display_word)
                print(f"\nConfirmed: {display_word}")
                last_confirmed  = display_word
                display_word    = "No sign"
                confidence      = 0.0
                last_probs      = np.zeros(len(ACTIONS))
                no_sign_counter = 0
                sequence.clear()
                smoother.reset()

        elif key == ord("c"):
            sequence.clear()
            sentence.clear()
            display_word    = "No sign"
            confidence      = 0.0
            last_probs      = np.zeros(len(ACTIONS))
            no_sign_counter = 0
            last_confirmed  = None
            smoother.reset()
            print("\nCleared.")

        elif key == ord("s"):
            speech_enabled[0] = not speech_enabled[0]
            print(f"\nSpeech: {'ON' if speech_enabled[0] else 'OFF'}")

print()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

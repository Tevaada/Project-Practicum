"""
collect_data.py — Collect sign language training sequences.

Controls:
  Q / ESC — stop early
  SPACE   — skip current sequence
"""
import os
import cv2
import numpy as np
from utilities import (
    DATA_PATH, ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH,
    mp_holistic, create_folders, mediapipe_detection,
    draw_styled_landmarks, extract_keypoints,
)

create_folders()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print(f"Collecting {NO_SEQUENCES} sequences x {SEQUENCE_LENGTH} frames")
print(f"Actions: {list(ACTIONS)}")
print("Q/ESC=stop   SPACE=skip sequence\n")


def wait_for_ready(cap, holistic, action, seq):
    """Live countdown before each sequence."""
    for countdown in range(3, 0, -1):
        deadline = cv2.getTickCount() + cv2.getTickFrequency()
        while cv2.getTickCount() < deadline:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # semi-transparent dark overlay drawn correctly
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

            h, w = image.shape[:2]
            cv2.putText(image, f"Action: {action.upper()}",
                        (w // 2 - 130, h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Sequence {seq + 1} / {NO_SEQUENCES}",
                        (w // 2 - 120, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(image, str(countdown),
                        (w // 2 - 30, h // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8, cv2.LINE_AA)

            cv2.imshow("Collect Data", image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return False
    return True


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    stopped = False

    for action in ACTIONS:
        if stopped:
            break
        print(f"\n--- Action: {action} ---")

        for seq in range(NO_SEQUENCES):
            if stopped:
                break

            last_npy = os.path.join(DATA_PATH, action, str(seq), f"{SEQUENCE_LENGTH - 1}.npy")
            if os.path.exists(last_npy):
                print(f"  seq {seq:02d} already exists, skipping")
                continue

            if not wait_for_ready(cap, holistic, action, seq):
                stopped = True
                break

            skip_seq = False
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    stopped = True
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                h, w = image.shape[:2]
                filled = int((frame_num + 1) / SEQUENCE_LENGTH * (w - 20))
                cv2.rectangle(image, (10, h - 20), (w - 10, h - 6), (60, 60, 60), -1)
                cv2.rectangle(image, (10, h - 20), (10 + filled, h - 6), (0, 220, 0), -1)

                cv2.putText(image,
                            f"{action}  |  seq {seq + 1}/{NO_SEQUENCES}  |  frame {frame_num + 1}/{SEQUENCE_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "REC", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("Collect Data", image)
                key = cv2.waitKey(10) & 0xFF

                np.save(
                    os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy"),
                    extract_keypoints(results),
                )

                if key in (ord("q"), 27):
                    stopped = True
                    break
                if key == ord(" "):
                    skip_seq = True
                    break

            if not skip_seq and not stopped:
                print(f"  seq {seq:02d} done")
            elif skip_seq:
                print(f"  seq {seq:02d} skipped")

cap.release()
cv2.destroyAllWindows()
print("\nStopped early." if stopped else "\nData collection complete.")

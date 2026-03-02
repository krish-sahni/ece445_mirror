import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

def main():
    # ---- download model once (simple, no extra deps) ----
    import os, urllib.request
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # ---- create landmarker ----
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe Tasks expects an mp.Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                # hand_landmarks is a list (one per detected hand)
                for hand in result.hand_landmarks:
                    # Draw landmarks (21 points)
                    for i, lm in enumerate(hand):
                        x_px, y_px = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)

                    # Index fingertip = landmark 8
                    idx = hand[8]
                    ix, iy = int(idx.x * w), int(idx.y * h)
                    cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
                    print(f"Index tip: norm=({idx.x:.3f}, {idx.y:.3f}) px=({ix}, {iy})", end="\r")

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Hand Tracking (MediaPipe Tasks)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

if __name__ == "__main__":
    main()
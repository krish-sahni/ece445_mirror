import cv2
import time
import mediapipe as mp

def main():
    # Mac-friendly backend; on Pi you can change to cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    # Pi-style perf settings (you can lower further if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # ---- "Lite" hand model ----
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,           # 0 = lite (fastest)
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)  # mirror view
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Index fingertip is landmark 8
                    idx = hand_landmarks.landmark[8]
                    ix, iy = int(idx.x * w), int(idx.y * h)
                    cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

                    # Print coords for cursor mapping
                    print(f"Index tip: norm=({idx.x:.3f},{idx.y:.3f}) px=({ix},{iy})", end="\r")

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("MediaPipe Hands (Pi-style lite)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
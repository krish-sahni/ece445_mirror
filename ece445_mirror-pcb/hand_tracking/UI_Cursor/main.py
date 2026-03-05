import cv2
from hand_tracker import HandTracker
from user_interface import HoverSelectUI

def main():
    # macOS-friendly backend. If black screen: try camera index 1 or remove backend flag.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    tracker = HandTracker(max_num_hands=1)
    ui = HoverSelectUI(dwell_seconds=1.5, smoothing_alpha=0.25, cursor_radius=10)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # mirror view (nice for mirror UX)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Hand tracking -> cursor update
            tip_norm = tracker.get_index_tip_norm(frame)
            ui.update_cursor_from_norm(tip_norm, w, h)

            # UI update + draw
            events = ui.update_and_draw(frame)
            for e in events:
                # For now, just print events. Later you can trigger actual actions.
                print(e)

            cv2.imshow("Hand UI Hover Select (Modular)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
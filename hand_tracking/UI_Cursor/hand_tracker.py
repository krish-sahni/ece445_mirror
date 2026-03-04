import os
import urllib.request
import cv2
import mediapipe as mp

TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
TASK_MODEL_PATH = "hand_landmarker.task"


class HandTracker:
    """
    Hand tracking abstraction.

    Primary output:
      get_index_tip_norm(bgr_frame) -> (x_norm, y_norm) in [0,1] or None

    Supports:
      - mp.solutions.hands (lite via model_complexity=0) if available
      - mediapipe.tasks HandLandmarker fallback if solutions is not available
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        prefer_solutions: bool = True,
        allow_tasks_fallback: bool = True,
    ):
        self.use_solutions = prefer_solutions and hasattr(mp, "solutions")
        self.hands = None
        self.landmarker = None

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if self.use_solutions:
            mp_hands = mp.solutions.hands
            # model_complexity=0 is the "lite" model (fastest)
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            if not allow_tasks_fallback:
                raise RuntimeError("mediapipe.solutions not available and tasks fallback disabled.")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not os.path.exists(TASK_MODEL_PATH):
                print("Downloading MediaPipe hand model (one-time)...")
                urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)

            base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self):
        if self.hands is not None:
            self.hands.close()
        if self.landmarker is not None:
            self.landmarker.close()

    def get_index_tip_norm(self, bgr_frame):
        """
        Return (x_norm, y_norm) for index fingertip (landmark 8) of the first detected hand.
        Returns None if no hand detected.
        """
        if self.use_solutions:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark[8]
                return (lm.x, lm.y)
            return None

        # Tasks API
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            idx = result.hand_landmarks[0][8]
            return (idx.x, idx.y)
        return None
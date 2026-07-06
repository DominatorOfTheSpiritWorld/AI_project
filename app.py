"""
ASL Sign Language Translator - Real-time Webcam App (MediaPipe Tasks API)
=========================================================================
Run train_model.py first, then: python app.py

CONTROLS:  Q/ESC = Quit  |  C = Clear  |  SPACE = Space  |  BACKSPACE = Delete
"""

import cv2
import numpy as np
import pickle
import time
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (5,9),(9,10),(10,11),(11,12),     # Middle
    (9,13),(13,14),(14,15),(15,16),   # Ring
    (13,17),(17,18),(18,19),(19,20),  # Pinky
    (0,17)                            # Palm base
]

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH            = "asl_model.pkl"
ENCODER_PATH          = "label_encoder.pkl"
HAND_LANDMARKER_MODEL = "hand_landmarker.task"

STABILITY_FRAMES      = 15
CONFIDENCE_THRESHOLD  = 0.6
LETTER_COOLDOWN       = 1.5

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        print("Model loaded.")
        return clf, le
    except FileNotFoundError:
        print("ERROR: Run train_model.py first.")
        exit(1)

# ─── LANDMARKER SETUP ────────────────────────────────────────────────────────

def create_landmarker():
    options = HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)

# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def extract_landmarks(landmarker, frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
  
    if not result.hand_landmarks:
        return None, None

    landmarks = result.hand_landmarks[0]
    wrist = landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    return np.array(features).reshape(1, -1), result

# ─── DRAW LANDMARKS ──────────────────────────────────────────────────────────

def draw_landmarks(frame, result):
    if not result or not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        connections = HAND_CONNECTIONS
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (0, 200, 100), 2)
        for pt in points:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 200, 100), 1)

# ─── DRAW UI ─────────────────────────────────────────────────────────────────

def draw_ui(frame, prediction, confidence, sentence, stable_progress):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    if prediction:
        label = prediction if prediction not in ["space", "del", "nothing"] else f"[{prediction}]"
        cv2.putText(frame, label, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 150), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{confidence*100:.0f}%", (160, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2, cv2.LINE_AA)

        bar_x1, bar_x2 = 220, w - 20
        cv2.rectangle(frame, (bar_x1, 52), (bar_x2, 74), (60, 60, 60), -1)
        filled = int((bar_x2 - bar_x1) * stable_progress)
        color = (0, 255, 150) if stable_progress < 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (bar_x1, 52), (bar_x1 + filled, 74), color, -1)
        cv2.putText(frame, "Hold...", (bar_x1 + 5, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hand detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2, cv2.LINE_AA)

    display = sentence[-50:] if len(sentence) > 50 else sentence
    cv2.putText(frame, display + "|", (15, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    for i, hint in enumerate(["Q: Quit", "C: Clear", "SPC: Space"]):
        cv2.putText(frame, hint, (w - 150, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def run():
    clf, le = load_model()
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)  # 0 = default built-in/USB webcam
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sentence        = ""
    last_letter     = None
    last_added_time = 0
    stable_queue    = deque(maxlen=STABILITY_FRAMES)

    print("ASL Translator running. Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        features, result = extract_landmarks(landmarker, rgb)

        prediction  = None
        confidence  = 0.0
        stable_frac = 0.0

        if features is not None:
            draw_landmarks(frame, result)

            proba      = clf.predict_proba(features)[0]
            top_idx    = np.argmax(proba)
            confidence = proba[top_idx]
            prediction = le.classes_[top_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                stable_queue.append(prediction)
            else:
                stable_queue.clear()
                prediction = None

            if len(stable_queue) == STABILITY_FRAMES and len(set(stable_queue)) == 1:
                stable_frac = 1.0
                now = time.time()
                if prediction != last_letter or (now - last_added_time) > LETTER_COOLDOWN:
                    if prediction == "space":
                        sentence += " "
                    elif prediction == "del":
                        sentence = sentence[:-1]
                    elif prediction != "nothing":
                        sentence += prediction.upper()
                    last_letter     = prediction
                    last_added_time = now
                    print(f"Detected: {prediction.upper()}  →  {sentence}")
            else:
                stable_frac = len(stable_queue) / STABILITY_FRAMES
        else:
            stable_queue.clear()

        draw_ui(frame, prediction, confidence, sentence, stable_frac)
        cv2.imshow("ASL Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            sentence = ""
        elif key == 32:
            sentence += " "
        elif key == 8:
            sentence = sentence[:-1]

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal sentence: {sentence}")

if __name__ == "__main__":
    run()

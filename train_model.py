"""
ASL Sign Language Classifier - Training Script (MediaPipe Tasks API)
=====================================================================
SETUP:
    pip install mediapipe==0.10.35 scikit-learn opencv-python numpy

DATASET:
    Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    Extract so you have: asl_alphabet_train/<letter>/<image>.jpg

USAGE:
    python train_model.py
"""

import os
import cv2
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATASET_PATH        = "archive/asl_alphabet_train/asl_alphabet_train"
MODEL_OUTPUT        = "asl_model.pkl"
ENCODER_OUTPUT      = "label_encoder.pkl"
MAX_IMAGES_PER_CLASS = 500

# Download the hand landmarker model:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
HAND_LANDMARKER_MODEL = "hand_landmarker.task"

# ─── SETUP LANDMARKER ────────────────────────────────────────────────────────

def create_landmarker():
    options = HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)

# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def extract_landmarks(landmarker, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    wrist = landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    return np.array(features)

# ─── DATASET LOADING ─────────────────────────────────────────────────────────

def load_dataset(landmarker, dataset_path):
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))
    print(f"Found {len(classes)} classes: {classes}\n")

    for label in classes:
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)[:MAX_IMAGES_PER_CLASS]
        success = 0

        for img_file in images:
            features = extract_landmarks(landmarker, os.path.join(class_dir, img_file))
            if features is not None:
                X.append(features)
                y.append(label)
                success += 1

        print(f"  [{label}] {success}/{len(images)} images processed")

    return np.array(X), np.array(y)

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)}...")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    return clf, le

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(HAND_LANDMARKER_MODEL):
        print("ERROR: hand_landmarker.task model file not found.")
        print("Download it from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("Place it in the same folder as this script.")
        exit(1)

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        print("Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        exit(1)

    print("=" * 50)
    print("  ASL Sign Language Model Trainer")
    print("=" * 50)

    landmarker = create_landmarker()
    X, y = load_dataset(landmarker, DATASET_PATH)

    print(f"\nTotal samples: {len(X)}")
    if len(X) == 0:
        print("ERROR: No landmarks extracted. Check your dataset.")
        exit(1)

    clf, le = train(X, y)

    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_OUTPUT, "wb") as f:
        pickle.dump(le, f)

    print(f"\nSaved: {MODEL_OUTPUT}, {ENCODER_OUTPUT}")
    print("Done! Now run: python app.py")

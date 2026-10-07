"""

ASL sign language translator (works sometimes)

Controls = q/esc for quit, spacebar for space (shocking), backspace for bacskpace (no way)

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



# tried using mp.solutions at first, but that didn't work for some reason, even tried to change the value in cv2.videocapture(0) or wtv from 0 to 1, -1 2 etc, and even                            # switched python models but that didn't work either, ended up switching to hand connections and hand landmarks, suubtracting the wrist paramters to generalize the landmarks



HAND_CONNECTIONS = [

    (0,1),(1,2),(2,3),(3,4),  # thumb

    (0,5),(5,6),(6,7),(7,8),  # index

    (5,9),(9,10),(10,11),(11,12), # middle

    (9,13),(13,14),(14,15),(15,16), # ring

    (13,17),(17,18),(18,19),(19,20), # pinky

    (0,17) # wrist/palm base

]



MODEL_PATH = "asl_model.pkl"

ENCODER_PATH = "label_encoder.pkl"

HAND_LANDMARKER_MODEL = "hand_landmarker.task"



STABILITY_FRAMES = 15 # how many frames in a row before it counts as a real letter, after testing 15 was the good spot and 0.6 confidence. 

CONFIDENCE_THRESHOLD = 0.6

LETTER_COOLDOWN = 1.5 # stops the same letter spamming if you just hold it





def load_model():

    #getting the trained model info basically, idk much about the training code as well, I used AI for it

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





def create_landmarker():

    options = HandLandmarkerOptions(

        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),

        running_mode=RunningMode.VIDEO, # video mode = smoother tracking vs IMAGE mode(tried image mode before, didnt work due to static individual images ig)

        num_hands=1,

        min_hand_detection_confidence=0.5,

        min_tracking_confidence=0.5,

        min_hand_presence_confidence=0.5,

    )

    return HandLandmarker.create_from_options(options)





def extract_landmarks(landmarker, frame_rgb):

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)



    if not result.hand_landmarks:

        return None, None



    landmarks = result.hand_landmarks[0]

    wrist = landmarks[0]



    # wrist is lm 0 so the ai knows what and where your hand is, as said above we subtract the wrist coords

    features = []

    for lm in landmarks:

        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])



    return np.array(features).reshape(1, -1), result





def draw_landmarks(frame, result):

    if not result or not result.hand_landmarks:

        return



    h, w = frame.shape[:2]

    for hand in result.hand_landmarks:

        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]



        for start, end in HAND_CONNECTIONS:

            cv2.line(frame, points[start], points[end], (0, 200, 100), 2)



        for pt in points:

            cv2.circle(frame, pt, 4, (255, 255, 255), -1)

            cv2.circle(frame, pt, 4, (0, 200, 100), 1) # colour of the landmarks





def draw_ui(frame, prediction, confidence, sentence, stable_progress):

    h, w = frame.shape[:2]



    # dark bar at top (the UI is mostly AI, as the arrays and stuff was really complex, but a lot of testing and debugging was done to fix webcam connecting issues

    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)



    # and bottom

    overlay2 = frame.copy()

    cv2.rectangle(overlay2, (0, h - 80), (w, h), (20, 20, 20), -1)

    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)



    if prediction:

        label = prediction if prediction not in ["space", "del", "nothing"] else f"[{prediction}]"

        cv2.putText(frame, label, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 150), 4, cv2.LINE_AA)

        cv2.putText(frame, f"{confidence*100:.0f}%", (160, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2, cv2.LINE_AA)



        # progress bar for the "hold it steady" thing

        bar_x1, bar_x2 = 220, w - 20

        cv2.rectangle(frame, (bar_x1, 52), (bar_x2, 74), (60, 60, 60), -1)

        filled = int((bar_x2 - bar_x1) * stable_progress)

        color = (0, 255, 150) if stable_progress < 1.0 else (0, 200, 255)

        cv2.rectangle(frame, (bar_x1, 52), (bar_x1 + filled, 74), color, -1)

        cv2.putText(frame, "Hold...", (bar_x1 + 5, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    else:

        cv2.putText(frame, "No hand detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2, cv2.LINE_AA)



    display = sentence[-50:] if len(sentence) > 50 else sentence

    cv2.putText(frame, display + "|", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)



    hints = ["Q: Quit", "C: Clear", "SPC: Space"]

    for i, hint in enumerate(hints):

        cv2.putText(frame, hint, (w - 150, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)





def run():

    clf, le = load_model()

    landmarker = create_landmarker()



    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        exit(1)



    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #size of UI



    sentence = ""

    last_letter = None

    last_added_time = 0

    stable_queue = deque(maxlen=STABILITY_FRAMES) # checks if the 15 stable frames were reached, and if they were all the same



    print("ASL Translator running. Press Q or ESC to quit.\n")



    while True:

        ret, frame = cap.read()

        if not ret:

            break



        # flip for mirror view, then convert since mediapipe wants rgb not bgr

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  



        features, result = extract_landmarks(landmarker, rgb)



        prediction = None

        confidence = 0.0

        stable_frac = 0.0



        if features is not None:

            draw_landmarks(frame, result)



            proba = clf.predict_proba(features)[0]

            top_idx = np.argmax(proba)

            confidence = proba[top_idx]

            prediction = le.classes_[top_idx]



            if confidence >= CONFIDENCE_THRESHOLD:

                stable_queue.append(prediction)

            else:

                stable_queue.clear()

                prediction = None



            # waiting until its sure of the letter before giving prediction (it's still wrong like 05% of the time)



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



                    last_letter = prediction

                    last_added_time = now

                    print(f"Detected: {prediction.upper()}  ->  {sentence}")

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

            sentence = sentence[:-1] # keys to clear, space bar etc etc.



    cap.release()

    cv2.destroyAllWindows()

    print(f"\nFinal sentence: {sentence}")





if __name__ == "__main__":

    run()

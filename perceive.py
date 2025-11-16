import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
import os

import easyocr
reader = easyocr.Reader(['en'])

# Monitor region to capture
monitor = {'top': 75, 'left': 650, 'width': 550, 'height': 650}

# Load YOLO model
model = YOLO("runs/detect/train_perceive4/weights/best.pt")

def run_detection(model):
    # Capture one frame from screen
    with mss() as sct:
        screenshot = np.array(sct.grab(monitor))

    # Convert to same format as training data
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Run YOLO on it
    results = model.predict(frame, imgsz=640, conf=0.25)

    troops_on_field = []
    troops_coords = []

    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        for cls, box in zip(classes, boxes):
            troops_on_field.append(model.names[int(cls)])
            troops_coords.append(box.tolist())

    return troops_on_field, troops_coords

def read_tower_hp(hps):
    
    with mss() as sct:
        full = np.array(sct.grab({'top': 75, 'left': 650, 'width': 550, 'height': 650}))
        full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)
    
    crops = [
        full[570:600, 100:160],   # tower 1
        full[570:600, 390:450],   # tower 2 
        full[80:110, 100:160],    # tower 3 
        full[80:110, 390:450],    # tower 4
    ]
    
    # Read raw OCR text
    results_raw = []
    for c in crops:
        txt = reader.readtext(c, detail=0)
        results_raw.append(txt[0] if txt else None)

    TOWER_MAX_HP = 3052.0
    final_hp_percents = []

    for i, raw in enumerate(results_raw):

        previous_percent = float(hps[i]) if hps[i] is not None else 1.0
        new_percent = previous_percent

        # ---- RULE: if tower is already dead skip all logic ----
        if previous_percent == 0:
            final_hp_percents.append(0)
            continue

        # ---- RULE: If OCR returned nothing, NaN, or invalid ----
        if raw is None or not raw.isdigit():
            # If tower had >20% last tick, keep old value
            if previous_percent > 0.2:
                new_percent = previous_percent
            else:
                # Tower was critically low; missing text likely means 0
                new_percent = 0
            final_hp_percents.append(new_percent)
            continue

        # ---- OCR returned a real number ----
        current_hp_raw = float(raw)
        observed_percent = current_hp_raw / TOWER_MAX_HP

        # RULE: HP cannot increase
        if observed_percent > previous_percent:
            new_percent = previous_percent
        else:
            new_percent = observed_percent

        final_hp_percents.append(new_percent)

    return final_hp_percents


def read_hand_cards():
    templates = {}
    for f in os.listdir("templates"):
        name = os.path.splitext(f)[0]  # strip .png/.jpg
        img = cv2.imread(os.path.join("templates", f), cv2.IMREAD_UNCHANGED)
        templates[name] = img

    # Define the hand region (adjust to your game)
    hand_region = {'top': 840, 'left': 725, 'width': 475, 'height': 150}
    card_width = hand_region['width'] // 4
    card_height = hand_region['height']

    # Capture screen
    with mss() as sct:
        screenshot = np.array(sct.grab(hand_region))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # ensure BGR

    # Crop the 4 cards
    hand_crops = [
        frame[0:card_height, i*card_width:(i+1)*card_width] for i in range(4)
    ]

    cards_in_hand = []

    for crop in hand_crops:
        # Ensure crop is BGR and uint8
        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)
        if crop.shape[2] == 4:  # BGRA -> BGR
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        best_score = 0
        best_match = None

        for name, tpl in templates.items():
            # Ensure template is BGR and uint8
            tpl_copy = tpl.copy()
            if tpl_copy.dtype != np.uint8:
                tpl_copy = tpl_copy.astype(np.uint8)
            if tpl_copy.shape[2] == 4:  # RGBA -> BGR
                tpl_copy = cv2.cvtColor(tpl_copy, cv2.COLOR_BGRA2BGR)

            result = cv2.matchTemplate(crop, tpl_copy, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_match = name

        cards_in_hand.append(best_match)

    return cards_in_hand

if __name__ == "__main__":
    pass
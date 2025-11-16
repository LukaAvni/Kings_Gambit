import mss
import cv2
import time
import os
import numpy as np

# Make folders
os.makedirs("vision/images/train", exist_ok=True)
os.makedirs("vision/images/val", exist_ok=True)

monitor = {'top': 75, 'left': 650, 'width': 550, 'height': 650}

MAX_IMAGES = 300

with mss.mss() as sct:
    counter = 1

    while counter <= MAX_IMAGES:

        screenshot = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Every 10th goes to val
        if counter % 10 == 0:
            out_path = f"vision/images/val/{counter:04d}.png"
        else:
            out_path = f"vision/images/train/{counter:04d}.png"

        cv2.imwrite(out_path, frame)
        print("Saved:", out_path)

        counter += 1
        time.sleep(1)  # capture once per second

print("DONE, collected all screenshots.")
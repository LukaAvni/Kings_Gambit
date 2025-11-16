import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
from pynput import keyboard

# Monitor region to capture
monitor = {'top': 75, 'left': 650, 'width': 550, 'height': 650}

# Load YOLO model
model = YOLO("runs/detect/train_perceive4/weights/best.pt")

# Flag to trigger prediction
predict_now = False

def on_press(key):
    global predict_now
    try:
        if key == keyboard.Key.enter:
            predict_now = True
    except AttributeError:
        pass

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

with mss() as sct:
    while True:
        # Capture the monitor region
        screenshot = np.array(sct.grab(monitor))

        # Convert to RGB in the same way as your training data (keeps the weird partial inversion)
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Show live preview
        cv2.imshow("Monitor Preview", frame)

        # Run prediction if Enter was pressed
        if predict_now:
            results = model.predict(frame, imgsz=640, conf=0.25)
            troops_on_field = []

            for r in results:
                classes = r.boxes.cls.cpu().numpy()  # class indices
                for cls in classes:
                    troops_on_field.append(model.names[int(cls)])

            print("Troops on field:", troops_on_field)
            predict_now = False

        # Exit if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()
listener.stop()
import mss
import cv2
import numpy as np

monitor = {'top': 75, 'left': 650, 'width': 550, 'height': 650} #for training

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        cv2.imshow("Preview - Press Q to exit", frame)

        # Press Q to close preview
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
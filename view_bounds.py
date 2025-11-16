import mss
import cv2
import numpy as np
import matplotlib.pyplot as plt

monitor = {'top': 0, 'left': 0, 'width': 1600, 'height': 1200} #for deck
#monitor = {'top': 75, 'left': 650, 'width': 550, 'height': 650} #for field

with mss.mss() as sct:
    screenshot = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Display using matplotlib
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Preview")
plt.axis('off')
plt.show()
from pynput.mouse import Controller, Button
import time
import os
import cv2

mouse = Controller()

def play(card, drop):
    if card == (None or 0):
        print("Decided not to play a card.")
        return
    
    DROP_ZONES = {
        1: (750, 500),   # left bridge
        2: (1100, 500),  # right bridge
        3: (900, 550),   # middle left
        4: (950, 550),   # middle right
    }
    
    CARD_SLOTS = {
        1: (800, 900),
        2: (920, 900),
        3: (1030, 900),
        4: (1130, 900),
    }

    card_x, card_y = CARD_SLOTS[card]
    drop_x, drop_y = DROP_ZONES[drop]

    #Move to card
    mouse.position = (card_x, card_y)
    time.sleep(0.05)

    #Click and drag to drop zone
    mouse.press(Button.left)
    time.sleep(0.05)
    mouse.position = (drop_x, drop_y)
    time.sleep(0.05)
    mouse.release(Button.left)

    print(f"Played card {card} to zone {drop}.")


def main():
    pass

if __name__ == "__main__":
    main()
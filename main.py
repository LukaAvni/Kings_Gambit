import perceive as see
import process as think
import prosecute as act
from ultralytics import YOLO

from datetime import datetime, timedelta
import time
import os
import sys

def main():
    model = YOLO("runs/detect/train_perceive4/weights/best.pt")
    hp = [1.0, 1.0, 1.0, 1.0]
    elixir = 0
    elixir_max = 10
    start_time = datetime.now()

    # Interval control
    regular_interval = 2.8
    fast_interval = 1.4
    game_minute = timedelta(seconds=60)
    
    while True:
        # Determine current interval based on game time
        elapsed = datetime.now() - start_time
        interval = fast_interval if elapsed >= game_minute else regular_interval

        # Increment elixir
        if elixir < elixir_max:
            elixir += 1

        # Read current game state
        troops = see.run_detection(model)
        hp = see.read_tower_hp(hp)
        hand = see.read_hand_cards()

        # Debug / log current state
        print(f"Elixir: {elixir}\nHP: {hp}\nTroops: {troops}\nHand: {hand}")

        # Brain chooses an action
        card_choice, drop_choice = think.brain(elixir, hp, troops, hand)

        # Execute the action
        act.play(card_choice, drop_choice)

        # Wait until next tick
        time.sleep(interval)
    
if __name__ == "__main__":
    main()
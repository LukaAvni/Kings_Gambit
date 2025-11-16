import torch
import perceive as see
import process as think
import prosecute as act
from ultralytics import YOLO

from datetime import datetime, timedelta
import time
import os
import sys
import threading

def main():
    model = YOLO("runs/detect/train_perceive4/weights/best.pt")
    hp = [1.0, 1.0, 1.0, 1.0]
    elixir = 10
    elixir_max = 10
    start_time = datetime.now()
    

    # Interval control
    regular_interval = 2.8
    fast_interval = 1.4
    game_minute = timedelta(seconds=60)
    
    CARD_COST = {
        "msno": 2,
        "mice": 1,
        "mlar": 1,
        "mhog": 4,
        "mcan": 3,
        "musk": 4,
        "mlog": 2,
        "mfir": 4
    }

    # Before the loop starts
    prev_hp = hp.copy()  # keep previous tower HP
    log_probs = []       # store log_probs for RL updates later
    rewards = []         # store rewards


    stop_flag = False

    def wait_for_enter():
        global stop_flag
        input("Press Enter to stop the game and train...\n")
        stop_flag = True

    threading.Thread(target=wait_for_enter, daemon=True).start()


    while not stop_flag:
        # Determine current interval based on game time
        elapsed = datetime.now() - start_time
        interval = fast_interval if elapsed >= game_minute else regular_interval

        # Increment elixir
        if elixir < elixir_max:
            elixir += 1

        # Read current game state
        troops, coords = see.run_detection(model)
        hp = see.read_tower_hp(hp)
        hand = see.read_hand_cards()

        # Debug / log current state
        print(f"Elixir: {elixir}\nHP: {hp}\nTroops: {troops}\nCoords: {coords}\nHand: {hand}")

        # Brain chooses an action
        state_tensor = encode(elixir, hp, troops, coords, hand)
        card_choice, drop_choice, log_prob = think.select_action(state_tensor)

        if card_choice>0 and elixir>CARD_COST[hand[card_choice-1]]:
            elixir -= CARD_COST[hand[card_choice-1]]
        else:
            card_choice = 0

        # Execute the action
        act.play(card_choice, drop_choice)

        #Track rewards
        reward = 0
        # Opponent towers (indices 2 and 3)
        for i, current in enumerate(hp[2:4], start=2):
            delta = prev_hp[i] - current
            if delta > 0:
                reward += 1 * delta
            if current <= 0 and prev_hp[i] > 0:
                reward += 5
        # Your towers (indices 0 and 1)
        for i, current in enumerate(hp[0:2]):
            delta = prev_hp[i] - current
            if delta > 0:
                reward -= 1 * delta
            if current <= 0 and prev_hp[i] > 0:
                reward -= 5

        # Store log_prob and reward
        log_probs.append(log_prob)
        rewards.append(reward)

        # Update previous HP
        prev_hp = hp.copy()

        # Wait until next tick
        time.sleep(interval)

    from process import update_policy
    update_policy(log_probs, rewards)
    print("Training complete. Weights saved.")

def encode(elixir, hp, troops, coords, hand):
    TROOP_TYPES = ["msno","mice","mlar","mhog","mcan","musk","mlog","mfir","osno","oice","olar","ohog","ocan","ousk","olog","ofir"]
    CARD_TYPES  = ["msno","mice","mlar","mhog","mcan","musk","mlog","mfir"]
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # 1. Elixir normalized
    elixir_tensor = torch.tensor([elixir / 10.0], dtype=torch.float32)

    # 2. Tower HPs
    hp_tensor = torch.tensor(hp, dtype=torch.float32)

    # 3. Troops one-hot
    troop_tensor_list = []
    for t in TROOP_TYPES:
        troop_tensor_list.append([1.0 if t in troops else 0.0])
    troop_tensor = torch.tensor(troop_tensor_list, dtype=torch.float32).flatten()

    # 4. Troop coordinates normalized
    max_troops = len(TROOP_TYPES)  # fixed max to match NN input
    coords = coords[:max_troops]   # truncate if too many troops
    # pad with zeros if fewer troops
    while len(coords) < max_troops:
        coords.append([0, 0, 0, 0])

    coord_tensor_list = []
    for x1, y1, x2, y2 in coords:
        coord_tensor_list.extend([x1/SCREEN_WIDTH, y1/SCREEN_HEIGHT, x2/SCREEN_WIDTH, y2/SCREEN_HEIGHT])
    coord_tensor = torch.tensor(coord_tensor_list, dtype=torch.float32)

    # 5. Hand cards one-hot
    hand_tensor_list = []
    for c in CARD_TYPES:
        hand_tensor_list.append([1.0 if c in hand else 0.0])
    hand_tensor = torch.tensor(hand_tensor_list, dtype=torch.float32).flatten()

    # Concatenate everything into a single tensor
    state_tensor = torch.cat([elixir_tensor, hp_tensor, troop_tensor, coord_tensor, hand_tensor])

    return state_tensor


if __name__ == "__main__":
    main()
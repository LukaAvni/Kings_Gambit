# Setup Instructions

## TRAINING THE VISION MODEL

### Data Engineering

**Gathering the data for the board:**

1. Run `view_bounds.py` and adjust `x-y` parameters to contain just the game window.
2. Run `train_populate_perceive.py`. This will populate screenshots of data. For each screenshot, you will have to label the data according to the YOLO model.
3. There are online tools for data annotation. Check out: [Roboflow](https://roboflow.com/)

**Folder structure after annotating:**

```
data/
    images/
        train/
            0001.png    # Screenshot of the whole game
        val/
            0001.png    # Same structure as train
    labels/
        train/
            0001.txt    # Formatted as: int_class coordinate_number c_number c_number c_number
                        # Note: the 5-tuple is per class and each class is a row in each screenshot
        val/
            0001.txt    # Same structure as train
```

### Training

1. Run `train_perceive.py`. It should store some weights.
   If the first steps were done correctly, the weights should be good.

### Running

1. The `perceive.py` file has all the functions related to data on screen.
   **Note:** Everything is currently fine-tuned to the aspect ratio of my monitor.
   You can use `view_bounds.py` to adjust it to yours.
2. After adjusting, running `run_detection()` inside `perceive.py` will tell you what cards are on the board.

## TRAINING THE BRAINS OF THE GAME

### Training

1. The brains are not trained through annotated data; they are trained via Reinforcement Learning.
   The model simply plays repeatedly to improve.

### Running

1. Since the model improves while playing, running and training happen simultaneously.

## ACTUALLY RUNNING THE AI

Simply run:

```bash
python main.py
```

The model will take control of your screen and play Clash Royale.
Make sure everything is in the correct spot that the AI will control by checking with `view_bounds.py`.

## ADDITIONAL DOCUMENTATION

* `main.py`: handles the 3 cores and the game logic.
* `perceive.py`: handles all the information retrieval from the board.
* `process.py`: handles all the logic and algorithms for improving at the game.
* `prosecute.py`: handles the interactions with the game, such as moving cards.
* `train_perceive.py`: processes all annotated images so that `perceive.py` is accurate.
* `train_populate_perceive.py`: generates screenshots (1 per second), useful when online data is unavailable.
* `view_bounds.py`: visualizes all the coordinate points in the game. Since we are humans and not machines, some back-and-forth is required.

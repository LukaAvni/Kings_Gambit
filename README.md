Setup instructions:

TRAINING THE VISION MODEL:

Data Engineering:

Gathering the data for the board:
1: Run view_bounds.py and adjust x-y parameters to contain just the game window.
2: Run train_populate_perceive.py, this will populate screenshots of data, then for each screeshot
        you will have to label the data according to the YOLO model.
3: There are online tools for data annotation, check out: https://roboflow.com/

Note that folderstructure should go as something like this after annotating:
data:
    images:
        train:
            0001.png --> Screenshot of the whole game
        val:
            same thing
    labels:
        train:
            0001.txt --> Formated as --> int_class coordinate_number c_number c_number c_number 
            Note the .txt 5-tuple is per class and each class is a row in each screenshot
        val:
            same thing

Training the Vision Model:
1: Run train_perceive.py and it should store some weights, if you did the first steps correctly they weights should be good.

Running The Vision Model:
1: The perceive.py file has all the functions related to data on screen. Please keep in mind everything is currently
fine tuned to the aspect ratio of my own monitor, but you can use view_bounds.py to adjust to yours. After adjusting,
running run_detection.py will tell you what cards are on the board.

TRAINING THE BRAINS OF THE GAME:


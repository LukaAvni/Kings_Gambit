Setup instructions:

TRAINING:

Data Engineering:

Gathering the data for the board:
1: Run view_bounds.py and adjust x-y parameters to contain just the game window.
2: Run train_populate_perceive.py, this will populate screenshots of data, then for each screeshot
        you will have to label the data according to the YOLO model.
3: There are online tools for data annotation, check out: https://roboflow.com/

Note that folderstructure should go as follows after annotating:
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
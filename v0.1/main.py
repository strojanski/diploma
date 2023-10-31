import os
import cv2 
import numpy as np


def read_raw():
    ear_data = os.listdir("./data/AWE")

    ear_imgs = {}
    for person in ear_data:
        ear_imgs[person] = [cv2.cvtColor(cv2.imread("./data/AWE/%s/%02d.png" % (person, i)), cv2.COLOR_BGR2RGB) for i in range(1, 11)]


mode = input("Model: ")
if mode == 'squeezenet' or mode == 1:
    # Call squeezenet.py
    pass

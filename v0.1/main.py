import os
import cv2 
import numpy as np

from squeezenet import squeezenet_train, squeezenet_test, squeezenet_preprocess


model = input("Model: ")
mode = int(input("Mode [1: train, 2: test, 3: preprocess]: "))

if model == 'squeezenet' or model == 1:
    # Call squeezenet.py
    if mode == 1:
        squeezenet_train()
    
    elif mode == 2:
        squeezenet_test()
        
    elif mode == 3:
        squeezenet_preprocess()

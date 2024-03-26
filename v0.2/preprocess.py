import copy
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os
import cv2


def resize_input(input_data: dict, tgt_size=224) -> dict:
    '''
        Input data: dictionary of images per person
        Output data: dictionary of images per person, resized to 2 tgt_size x 2 tgt_size
    '''

    resized_data = copy.deepcopy(input_data)
    for key in input_data.keys():   # For each person
        for i in range(len(input_data[key])):   # For each image
            
            # Resize into 224x448
            resized_data[key][i] = np.array(Image.fromarray(resized_data[key][i]).resize((int(.5*tgt_size), tgt_size)))
            
            # Convert to grayscale
            # resized_data[key][i] = np.stack([np.mean(resized_data[key][i], axis=2)] * 3, axis=2)
            
            # Zero pad to make it square
            pad = np.abs(resized_data[key][i].shape[1] - resized_data[key][i].shape[0]) // 2
            resized_data[key][i] = np.pad(resized_data[key][i], ((0, 0), (pad, pad), (0, 0)), 'constant')    
            
            # Normalize
            resized_data[key][i] = resized_data[key][i].astype(np.float32) / 255
                        
                        
            # train - rotation & contrast brightness, saturation hue, shear, randomcrop
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            resized_data[key][i] = preprocess(resized_data[key][i])
                        
    return resized_data

def train_test_split(input_data: dict, test_size=0.3) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X_train, X_test = {}, {}

    # X data split, y = person
    for person, imgs in input_data.items():
        np.random.shuffle(imgs)
        X_train[person], X_test[person] = imgs[:8], imgs[8:]
        
    y_train = np.array([int(person) for person in X_train.keys()])
    y_test = np.array([int(person) for person in X_test.keys()])
    X_train = np.array([np.array([np.array(img) for img in person]) for person in X_train.values()])
    X_test = np.array([np.array([np.array(img) for img in person]) for person in X_test.values()])

    # Reshape (100, 7) -> (700, 1)
    y_train = np.array([label for label in y_train for _ in range(8)]).astype(np.int64)
    y_test = np.array([label for label in y_test for _ in range(2)]).astype(np.int64)
    X_train = np.array([img for person in X_train for img in person]).astype(np.float32)
    X_test = np.array([img for person in X_test for img in person]).astype(np.float32)        
        
    return X_train, X_test, y_train, y_test


def read_raw():
    ear_data = os.listdir("./data/AWE")

    ear_imgs = {}
    for person in ear_data:
        ear_imgs[person] = [cv2.cvtColor(cv2.imread("./data/AWE/%s/%02d.png" % (person, i)), cv2.COLOR_BGR2RGB) for i in range(1, 11)]

    return ear_imgs

if __name__ == '__main__':
    pass

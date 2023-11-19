import copy
import numpy as np
from PIL import Image
# from torchvision import transforms
from torchvision.transforms import v2 as transforms
import torchvision
import torch
import os
import cv2
import matplotlib.pyplot as plt


torchvision.disable_beta_transforms_warning()


def resize_input(input_data: np.ndarray, tgt_size=224, mode="train") -> dict:
    '''
        Input data: arary of images
        Output data: array of images, resized to 2 tgt_size x 2 tgt_size
    '''
    torchvision.disable_beta_transforms_warning()
    
    # train - rotation & contrast brightness, saturation hue, shear, randomcrop
    preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomShortestSize(200),
        # transforms.ElasticTransform(),
        # transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.RandomPerspective(distortion_scale=.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        # transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(brightness=.1, contrast=0.1, saturation=.1, hue=.01),
        transforms.ConvertImageDtype(torch.float32),
        # transforms.Normalize(mean=[0.4026756, 0.40258485, 0.40231562], std=[0.26870993, 0.268518, 0.2680013]),
        transforms.Resize([224, 112]),
        transforms.Pad([56, 0])
    ])
    
    if mode == "test":
        preprocess = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4026756, 0.40258485, 0.40231562], std=[0.26870993, 0.268518, 0.2680013]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize([224, 112]),
            transforms.Pad([56, 0])
        ])
            
    tensor = transforms.ToTensor()


    for i, img in enumerate(input_data):   # For each image
        img = img / 255.0  # Normalize the image
        img = tensor(img)
        
        # img = tensor(img)
        img = preprocess(img)
                
        input_data[i] = img

                    
                                            
    return input_data

def train_test_split(input_data: dict, test_size=0.3) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X_train, X_test = [], []
    y_train, y_test = [], []

    # X data split, y = person
    for person, imgs in input_data.items():
        # np.random.shuffle(imgs)
        X_train.extend(imgs[:8])
        X_test.extend(imgs[8:])
        y_train.extend([int(person)] * len(imgs[:8]))
        y_test.extend([int(person)] * len(imgs[8:]))
        
    y_train = np.array(list(y_train)) - 1
    y_test = np.array(list(y_test)) - 1
                
                
    return X_train, X_test, y_train, y_test


def read_raw():
    ear_data = os.listdir("./data/AWE")

    ear_imgs = {}
    for person in ear_data:
        imgs = os.listdir("./data/AWE/%s" % person)
        try:
            ear_imgs[person] = [cv2.cvtColor(cv2.imread(f"./data/AWE/{person}/{img}"), cv2.COLOR_BGR2RGB) for img in imgs]
        except Exception as e:
            print(e)
    return ear_imgs

if __name__ == '__main__':
    pass

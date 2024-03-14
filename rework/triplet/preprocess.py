import copy
import os

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
# from torchvision import transforms
from torchvision.transforms import v2 as transforms

torchvision.disable_beta_transforms_warning()



def resize_input(input_data: np.ndarray, tgt_size=64, mode="train"):
    """
    Input data: arary of images
    Output data: array of images, resized to 2 tgt_size x 2 tgt_size
    """
    torchvision.disable_beta_transforms_warning()

    # train - rotation & contrast brightness, saturation hue, shear, randomcrop
    preprocess = transforms.Compose(
        [
            transforms.Resize(tgt_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize([tgt_size, tgt_size//2], interpolation=Image.BICUBIC),
            transforms.Pad([tgt_size//4, 0]),
        ]
    )

    if mode == "test":
        preprocess = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize([tgt_size, tgt_size//2], interpolation=Image.BICUBIC),
                transforms.Pad([tgt_size//4, 0]),
            ]
        )

    tensor = transforms.ToTensor()

    for i, img in enumerate(input_data):  # For each image
        img = tensor(img)

        img = preprocess(img)

        input_data[i] = img

    return input_data


# Usage example:
# processed_data = resize_input(input_data, tgt_size=224, mode="train")


def train_test_split(input_data: dict, test_ssize=0.3):
    X_train, X_test = [], []
    y_train, y_test = [], []

    # X data split, y = person
    for person, imgs in input_data.items():
        n_imgs = len(imgs)
        train_size = int(n_imgs * (1 - test_ssize))
        
        # Shuffle images
        np.random.shuffle(imgs)
        
        X_train.extend(imgs[:train_size])
        X_test.extend(imgs[train_size:])
        y_train.extend([int(person)] * len(imgs[:train_size]))
        y_test.extend([int(person)] * len(imgs[train_size:]))

    y_train = np.array(list(y_train)) - 1
    y_test = np.array(list(y_test)) - 1

    return X_train, X_test, y_train, y_test


def read_raw(train_subjects, path="../UERC"):
    ear_data = os.listdir(path)

    ear_imgs = {}
    for person in ear_data:
        if person not in train_subjects:
            continue
        
        imgs = os.listdir("./data/AWE/%s" % person)
        try:
            ear_imgs[person] = [
                cv2.cvtColor(
                    cv2.imread(f"./data/AWE/{person}/{img}"), cv2.COLOR_BGR2RGB
                )
                for img in imgs
            ]
        except Exception as e:
            print(e)
    return ear_imgs


if __name__ == "__main__":
    pass

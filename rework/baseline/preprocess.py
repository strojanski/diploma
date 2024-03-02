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


def resize_input(input_data: np.ndarray, tgt_size=224, mode="train") -> dict:
    """
    Input data: arary of images
    Output data: array of images, resized to 2 tgt_size x 2 tgt_size
    """
    torchvision.disable_beta_transforms_warning()

    # train - rotation & contrast brightness, saturation hue, shear, randomcrop
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            # transforms.RandomShortestSize(200),
            # transforms.ElasticTransform(),
            # transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomRotation(degrees=25),
            # transforms.RandomPerspective(distortion_scale=.15),
            # transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            # transforms.GaussianBlur(kernel_size=3),
            # transforms.ColorJitter(
            #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01
            # ),
            transforms.ConvertImageDtype(torch.float32),
            # transforms.Normalize(mean=[0.4026756, 0.40258485, 0.40231562], std=[0.26870993, 0.268518, 0.2680013]),
            transforms.Resize([224, 112]),
            transforms.Pad([56, 0]),
        ]
    )

    if mode == "test":
        preprocess = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize([224, 112]),
                transforms.Pad([56, 0]),
            ]
        )

    tensor = transforms.ToTensor()

    for i, img in enumerate(input_data):  # For each image
        img = img / 255.0  # Normalize the image
        img = tensor(img)

        img = preprocess(img)

        input_data[i] = img

    return input_data


def train_test_split(input_data: dict, test_ssize=0.3):
    X_train, X_test = [], []
    y_train, y_test = [], []

    # X data split, y = person
    for person, imgs in input_data.items():
        n_imgs = len(imgs)
        train_size = int(n_imgs * (1 - test_ssize))
        test_size = int(n_imgs * test_ssize)
        
        X_train.extend(imgs[:train_size])
        X_test.extend(imgs[train_size:])
        y_train.extend([int(person)] * len(imgs[:-test_size]))
        y_test.extend([int(person)] * len(imgs[-test_size:]))

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

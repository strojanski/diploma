import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

# from torchvision import transforms
from torchvision.transforms import v2 as transforms

torchvision.disable_beta_transforms_warning()


def resize_input(input_data: np.ndarray, tgt_size=64, mode="train") -> dict:
    """
    Input data: arary of images
    Output data: array of images, resized to 2 tgt_size x 2 tgt_size
    """
    torchvision.disable_beta_transforms_warning()

    # train - rotation & contrast brightness, saturation hue, shear, randomcrop
    preprocess = transforms.Compose(
        [
            transforms.Resize(tgt_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize([tgt_size, tgt_size // 2], interpolation=Image.BICUBIC),
            transforms.Pad([tgt_size // 4, 0]),
        ]
    )

    if mode == "test":
        preprocess = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(
                    [tgt_size, tgt_size // 2], interpolation=Image.BICUBIC
                ),
                transforms.Pad([tgt_size // 4, 0]),
            ]
        )

    tensor = transforms.ToTensor()

    for i, img in enumerate(input_data):  # For each image
        img = img / 255.0  # Normalize the image
        img = tensor(img)

        img = preprocess(img)

        input_data[i] = img

    return input_data


def train_test_split(
    input_data: dict, test_size=0.3
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X_train, X_test = [], []
    y_train, y_test = [], []

    # X data split, y = person
    for person, imgs in input_data.items():
        X_train.extend(imgs[:-2])
        X_test.extend(imgs[-2:])
        y_train.extend([int(person)] * len(imgs[:-2]))
        y_test.extend([int(person)] * len(imgs[-2:]))

    y_train = np.array(list(y_train)) - 1
    y_test = np.array(list(y_test)) - 1

    return X_train, X_test, y_train, y_test


def read_raw():
    ear_data = os.listdir("./data/UERC")

    ear_imgs = {}
    n_imgs = 0
    for c, person in enumerate(ear_data):
        imgs = os.listdir("./data/UERC/%s" % person)

        try:
            ear_imgs[person] = [
                cv2.cvtColor(
                    cv2.imread(f"./data/UERC/{person}/{img}"), cv2.COLOR_BGR2RGB
                )
                for img in imgs
            ]
            n_imgs += len(ear_imgs[person])

            if c % 10 == 0:
                print(c, n_imgs)

        except Exception as e:
            print(e)
    return ear_imgs


if __name__ == "__main__":
    pass

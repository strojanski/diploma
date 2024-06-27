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
            transforms.RandomHorizontalFlip(p=0.3),
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

if __name__ == "__main__":
    pass

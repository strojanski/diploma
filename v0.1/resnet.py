import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

import cv2
import os

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import io

from preprocess import resize_input, train_test_split, read_raw
from ear_dataset import EarDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def get_train_data(train_dataset, batch_size=32):
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader


def get_test_data(test_dataset, batch_size=32):
    trainloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader


def get_resnet():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    # model = resnet50(weights=ResNet50_Weights.DEFAULT)

    return model


def train_resnet(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 50
    model = model.to(device)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for img_batch, label_batch in dataloader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(img_batch)
            loss = criterion(output, label_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loss_history.append(loss.item())

        print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss}")

    plt.plot(loss_history)
    plt.show()

    return model


def test_resnet(model, dataloader):
    model.eval()

    predictions = []
    ys = []
    model = model.to(device)
    with torch.no_grad():  # Disable gradiant calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            ys.extend(labels.tolist())
            predictions.extend(preds.cpu().tolist())

    print(predictions, "\n", ys)
    print(np.array(predictions).shape, np.array(ys).shape)
    return accuracy_score(ys, predictions)


def resnet_train():

    # Read in input data
    input_data = read_raw()

    train_dataset = torch.load("data/train_dataset.pt")

    # Create train data loader
    train_dataloader = get_train_data(train_dataset)

    # Get model and modify classifier
    model = get_resnet()
    num_features = model.fc.in_features
    num_classes = 101

    model.fc = nn.Linear(num_features, num_classes)

    # Train model
    model = train_resnet(model, train_dataloader)

    torch.save(model, "models/resnet.pt")


def resnet_test():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(device)

    input_data = read_raw()

    test_dataset = torch.load("data/test_dataset.pt")
    train_dataset = torch.load("data/train_dataset.pt")

    test_dataloader = get_test_data(test_dataset)
    train_dataloader = get_train_data(train_dataset)

    data_iter = iter(test_dataloader)
    batch = next(data_iter)
    test_imgs, test_labels = batch

    model = torch.load("models/resnet.pt")

    score = test_resnet(model, test_dataloader)
    print(f"Accuracy: {score * 100}%")

    test = test_imgs.to(device)
    outputs = model(test)

    for i, output in enumerate(outputs):
        pred, truth = torch.argmax(output), test_labels[i]
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(test_imgs[i].permute(1, 2, 0))
        ax[0].set_title(f"Truth: {truth}")

        ax[1].imshow(input_data["{:03d}".format(pred)][0])
        ax[1].set_title(f"Pred: {pred}")

        plt.show()


def resnet_preprocess():

    input_data = read_raw()

    # Resize input data
    input_data = resize_input(input_data, tgt_size=224)

    # plt.imshow(input_data["001"][0], cmap='gray')
    # plt.show()
    # print(input_data["001"][0].shape)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(input_data)

    # Create train data set
    train_dataset = EarDataset(X_train, y_train)
    test_dataset = EarDataset(X_test, y_test)

    torch.save(train_dataset, "data/train_dataset.pt")
    torch.save(test_dataset, "data/test_dataset.pt")


if __name__ == "__main__":
    mode = "other"
    mode = "preprocess"
    mode = "train"
    mode = "test"

    if mode == "preprocess":
        resnet_preprocess()

    if mode == "train":
        resnet_train()

    if mode == "test":
        resnet_test()

    if mode == "other":
        train_dataset = torch.load("data/train_dataset.pt")
        test_dataset = torch.load("data/test_dataset.pt")

        label_range = (train_dataset.labels.min(), train_dataset.labels.max())
        print(label_range)

        test_dataloader = get_test_data(torch.load("data/test_dataset.pt"))
        train_dataloader = get_train_data(torch.load("data/train_dataset.pt"))

        # iterator = iter(train_dataloader)
        # batch = next(iterator)

        # print(np.array(batch).shape)

        # imgs, labels = batch
        # print(imgs.shape, labels)

        # for i in range(10):
        #     img, label = imgs[i], labels[i]
        #     print(label)
        #     plt.imshow(img.permute(1, 2, 0))
        #     plt.show()

        # print(sample_label[0])
        # plt.imshow(sample_img[0].permute(1, 2, 0))
        # plt.show()

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score
from torch import nn
from torch.nn import Softmax
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms as T

from ear_dataset import EarDataset

parser = argparse.ArgumentParser(description="Person recognition on ear dataset.")
parser.add_argument(
    "--mode", type=str, default="train", help="Options: train, test, preprocess, other"
)
parser.add_argument("--bs", type=int, default=16, help="Batch size")


def get_train_data(train_dataset, batch_size=16):
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    return trainloader


def get_test_data(test_dataset, batch_size=16):
    trainloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    return trainloader


def get_model(name="resnet"):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet50", pretrained=True
    )  # test 50

    if name == "squeezenet":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=False
        )

    elif name == "densenet":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "densenet161", pretrained=False
        )

    elif name == "resnext":
        model = models.resnext50_32x4d()

    elif name == "inception":
        models.inception_v3()

    elif name == "alexnet":
        model = models.alexnet()

    elif name == "wideresnet":
        model = models.wide_resnet101_2()

    elif name == "googlenet":
        model = models.googlenet(pretrained=True)

    return model


def train(model):
    torchvision.disable_beta_transforms_warning()

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1, verbose=True
    )
    epochs = 100
    model = model.to(device)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for img_batch, label_batch in train_dataloader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = None
            if model_name == "googlenet":
                output = model(img_batch)
            else:
                output = model(img_batch)
            loss = criterion(output, label_batch)

            # Backward pass
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            loss_history.append(loss.item())

            if epoch % 10 == 0:
                torch.save(model, f"models/{model_name}.pt")

        scheduler.step()

        print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss}")

    plt.plot(loss_history)
    plt.show()

    return model


def calculate_metrics(predictions, ys, label):
    TP = np.sum((np.array(ys) == label) & (np.array(predictions) == label))
    TN = np.sum((np.array(ys) != label) & (np.array(predictions) != label))
    FP = np.sum((np.array(ys) != label) & (np.array(predictions) == label))
    FN = np.sum((np.array(ys) == label) & (np.array(predictions) != label))
    return TP, TN, FP, FN


def test(model):
    model.eval()

    predictions = []
    ys = []
    model = model.to(device)
    with torch.no_grad():  # Disable gradiant calculation
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            ys.extend(labels.tolist())
            predictions.extend(preds.cpu().tolist())

    print(np.array(predictions).shape, np.array(ys).shape)

    acc = 0
    recall = 0
    for label in set(ys):
        TP, TN, FP, FN = calculate_metrics(predictions, ys, label)
        print(f"Class {label}: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        acc += (TP + TN) / (TP + FP + TN + FN)
        recall += TP / (TP + FN)

    recall /= len(set(ys))
    print("Recall: ", recall)
    print("Accuracy score: ", acc / len(set(ys)))
    # print(f"Label: {label}, Accuracy: {, )}")
    # print("Precision score: ", precision_score(ys, predictions, average=None))
    print("Precision score: ", precision_score(ys, predictions, average="micro"))

    return accuracy_score(ys, predictions)


def get_mean_std(dataset):
    means = []
    stds = []
    for img in dataset:
        means.append(torch.mean(img))
        stds.append(torch.std(img))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

    return means, stds


if __name__ == "__main__":
    torchvision.disable_beta_transforms_warning()
    warnings.filterwarnings("ignore")

    torch.cuda.empty_cache()

    mode = parser.parse_args().mode
    batch_size = parser.parse_args().bs

    from preprocess import read_raw, resize_input, train_test_split

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(device)

    model_name = "squeezenet"
    model_name = "densenet"
    model_name = "vgg"
    model_name = "resnet"
    model_name = "inception"
    model_name = "alexnet"
    model_name = "wideresnet"
    model_name = "resnext"
    model_name = "googlenet"

    print(model_name)

    # Read in input data
    input_data = read_raw()

    if mode == "preprocess":
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(input_data)

        # Preprocess data
        X_train = resize_input(X_train, tgt_size=224, mode="train")
        X_test = resize_input(X_test, tgt_size=224, mode="test")

        fig, ax = plt.subplots(3, 3)
        for i in range(1, 10):
            print(X_train[i].shape)
            img = X_train[i + 10 * i].permute(1, 2, 0)
            ax[i // 3 - 1][i % 3].imshow(img)
        plt.show()
        print(input_data["001"][0].shape)

        # Create train data set
        train_dataset = EarDataset(X_train, y_train)
        test_dataset = EarDataset(X_test, y_test)

        torch.save(train_dataset, "data/train_dataset.pt")
        torch.save(test_dataset, "data/test_dataset.pt")

    if mode == "train":
        train_dataset = torch.load("data/train_dataset.pt")
        train_dataset.labels_to_long()

        # Create train data loader
        train_dataloader = get_train_data(train_dataset, batch_size=batch_size)

        m = "new"
        m = "old"

        model = None
        if m == "new":
            # Get model and modify classifier
            model = get_model(model_name)

            print(
                f"{len(set(train_dataset.labels)), min(train_dataset.labels), max(train_dataset.labels)}"
            )

            n_classes = train_dataset.get_n_classes()
            print(f"Number of classes: {n_classes}")

            # Change structure of classifier
            if model_name == "squeezenet":
                model.classifier[1] = nn.Conv2d(
                    512, n_classes, kernel_size=(1, 1), stride=(1, 1)
                )
            elif (
                model_name == "resnet"
                or model_name == "inception"
                or model_name == "resnext"
                or model_name == "wideresnet"
                or model_name == "googlenet"
            ):
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, n_classes)
            elif model_name == "densenet":
                num_features = model.classifier.in_features
                model.classifier = nn.Linear(num_features, n_classes)
            elif model_name == "alexnet":
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_features, n_classes)
        else:
            model = torch.load(f"models/{model_name}.pt")
            model.train()

        # Train model
        model = train(model)

        torch.save(model, f"models/{model_name}.pt")

    if mode == "test":
        test_dataset = torch.load("data/test_dataset.pt")
        train_dataset = torch.load("data/train_dataset.pt")

        test_dataloader = get_test_data(test_dataset, batch_size=batch_size)
        train_dataloader = get_train_data(train_dataset, batch_size=batch_size)

        test_imgs, test_labels = next(iter(test_dataloader))

        model = torch.load(f"models/{model_name}.pt")

        score = test(model)
        print(f"Accuracy: {score*100}%")

        test_ = test_imgs.to(device)
        outputs = model(test_)

        for i, output in enumerate(outputs[:5]):
            # print(output)
            # print(torch.max(output), torch.argmax(output), Softmax(output))
            print(torch.argmax(torch.nn.functional.softmax(output)))
            # output = Softmax(output) # Get probabilites
            pred, truth = torch.argmax(output), test_labels[i]
            fig, ax = plt.subplots(3, 3)

            ax[0][0].imshow(test_imgs[i].permute(1, 2, 0))
            ax[0][0].set_title(f"Truth: {truth}")

            train_data = train_dataset.get_class_data(pred)

            for i in range(1, 9):
                try:
                    ax[i // 3][i % 3].imshow(train_data[i].permute(1, 2, 0))
                    ax[0][2].set_title(f"Pred: {pred}")
                except IndexError as e:
                    print(e)

            plt.show()

    if mode == "other":
        # X_train, X_test, y_train, y_test = train_test_split(input_data)

        train_dataset = torch.load("data/train_dataset.pt")
        test_dataset = torch.load("data/test_dataset.pt")

        test_dataloader = get_test_data(test_dataset, batch_size=batch_size)
        train_dataloader = get_train_data(train_dataset, batch_size=batch_size)

        train_iter = iter(train_dataloader)
        test_iter = iter(test_dataloader)

        train_batch = next(train_iter)
        test_batch = next(test_iter)

        train_imgs, train_labels = train_batch
        test_imgs, test_labels = test_batch

        for i in range(10):
            fig, ax = plt.subplots(1, 2)

            ax[0].imshow(train_imgs[i].permute(1, 2, 0))
            ax[0].set_title(f"Train: {train_labels[i]}")

            ax[1].imshow(test_imgs[i].permute(1, 2, 0))
            ax[1].set_title(f"Test: {test_labels[i]}")

            plt.show()

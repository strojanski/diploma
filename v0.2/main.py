import torch
from torch import nn

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from preprocess import resize_input, train_test_split, read_raw
from ear_dataset import EarDataset


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


def get_model(name="resnet"):
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", pretrained=False
    )  # test 50

    if name == "squeezenet":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=False
        )

    elif name == "densenet":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "densenet161", pretrained=False
        )

    return model


def train(model):
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 20
    model = model.to(device)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for img_batch, label_batch in train_dataloader:
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


def test(model):
    model.eval()

    predictions = []
    ys = []
    model = model.to(device)
    with torch.no_grad():  # Disable gradiant calculation
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            ys.extend(labels.tolist())
            predictions.extend(preds.cpu().tolist())

    print(predictions, "\n", ys)
    print(np.array(predictions).shape, np.array(ys).shape)
    return accuracy_score(ys, predictions)


if __name__ == "__main__":

    mode = "other"
    mode = "preprocess"
    mode = "train"
    mode = "test"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(device)

    model_name = "squeezenet"
    model_name = "densenet"
    model_name = "resnet"
    print(model_name)

    # Read in input data
    input_data = read_raw()

    if mode == "preprocess":

        # Resize input data
        input_data = resize_input(input_data, tgt_size=224)

        plt.imshow(input_data["001"][0], cmap="gray")
        plt.show()
        print(input_data["001"][0].shape)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(input_data)

        # Create train data set
        train_dataset = EarDataset(X_train, y_train)
        test_dataset = EarDataset(X_test, y_test)

        torch.save(train_dataset, "data/train_dataset.pt")
        torch.save(test_dataset, "data/test_dataset.pt")

    if mode == "train":
        train_dataset = torch.load("data/train_dataset.pt")

        # Create train data loader
        train_dataloader = get_train_data(train_dataset)

        # Get model and modify classifier
        model = get_model(model_name)

        n_classes = 101

        # Change structure of classifier
        if model_name == "squeezenet":
            model.classifier[1] = nn.Conv2d(
                512, n_classes, kernel_size=(1, 1), stride=(1, 1)
            )
        elif model_name == "resnet":
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_classes)
        elif model_name == "densenet":
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, n_classes)

        # Train model
        model = train(model)

        torch.save(model, f"models/{model_name}.pt")

    if mode == "test":
        test_dataset = torch.load("data/test_dataset.pt")
        train_dataset = torch.load("data/train_dataset.pt")

        test_dataloader = get_test_data(test_dataset)
        train_dataloader = get_train_data(train_dataset)

        data_iter = iter(test_dataloader)
        batch = next(data_iter)
        test_imgs, test_labels = batch

        model = torch.load(f"models/{model_name}.pt")

        score = test(model)
        print(f"Accuracy: {score*100}%")

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

    if mode == "other":

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

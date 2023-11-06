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

from preprocess import resize_input
from ear_dataset import EarDataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def read_raw():
    ear_data = os.listdir("./data/AWE")

    ear_imgs = {}
    for person in ear_data:
        ear_imgs[person] = [cv2.cvtColor(cv2.imread("./data/AWE/%s/%02d.png" % (person, i)), cv2.COLOR_BGR2RGB) for i in range(1, 11)]

    return ear_imgs


def train_test_split(input_data: dict, test_size=0.3) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X_train, X_test = {}, {}

    # X data split, y = person
    for person, imgs in input_data.items():
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


def get_train_data(train_dataset, batch_size=32):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def get_test_data(test_dataset, batch_size=32):
    trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def get_squeezenet():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', weights="SqueezeNet1_1_Weights.DEFAULT")
    return model


def train_squeezenet(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 70
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


def test_squeezenet(model):
    model.eval()
    
    predictions = []
    ys = []
    model = model.to(device)
    with torch.no_grad():   # Disable gradiant calculation
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            ys.extend(labels.tolist())
            predictions.extend(preds.cpu().tolist())

    print(predictions, "\n", ys)
    print(np.array(predictions).shape, np.array(ys).shape)
    return accuracy_score(ys, predictions)



    
def squeezenet_train():
    
    # Read in input data
    input_data = read_raw()
    
    train_dataset = torch.load("data/train_dataset.pt")
    
    # Create train data loader
    train_dataloader = get_train_data(train_dataset)   
    
    # Get model and modify classifier
    model = get_squeezenet()
    n_classes = 101 
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))

    # Train model
    model = train_squeezenet(model, train_dataloader)

    torch.save(model, "models/squeezenet.pt")
    
def squeezenet_test():

    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    
    input_data = read_raw()
    
    test_dataset = torch.load("data/test_dataset.pt")
    train_dataset = torch.load("data/train_dataset.pt")
    
    test_dataloader = get_test_data(test_dataset)
    train_dataloader = get_train_data(train_dataset)

    data_iter = iter(test_dataloader)
    batch = next(data_iter)
    test_imgs, test_labels = batch 
    
    model = torch.load("models/squeezenet.pt")        
    
    score = test_squeezenet(model)
    print(f"Accuracy: {score}%")
    
    test = test_imgs.to(device)
    outputs = model(test)
    
    for i, output in enumerate(outputs):
        pred, truth = torch.argmax(output), test_labels[i]
        fig, ax = plt.subplots(1,2)
        
        ax[0].imshow(test_imgs[i].permute(1, 2, 0))
        ax[0].set_title(f"Truth: {truth}")
        
        ax[1].imshow(input_data["{:03d}".format(pred)][0])
        ax[1].set_title(f"Pred: {pred}")

        plt.show()

def squeezenet_preprocess():
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
        
        
if __name__ == '__main__':
    mode = 'other'
    mode = 'preprocess'
    mode = 'train'
    mode = 'test'
    
    if mode == 'preprocess':
        squeezenet_preprocess()
        
    if mode == 'train':
        squeezenet_train()
        
    if mode == 'test':
        squeezenet_test()
        
    if mode == 'other':    
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
        

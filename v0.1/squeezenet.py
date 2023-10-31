import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

import cv2
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import io

from preprocess import resize_input
from ear_dataset import EarDataset

def read_raw():
    ear_data = os.listdir("./data/AWE")

    ear_imgs = {}
    for person in ear_data:
        ear_imgs[person] = [cv2.cvtColor(cv2.imread("./data/AWE/%s/%02d.png" % (person, i)), cv2.COLOR_BGR2RGB) for i in range(1, 11)]

    return ear_imgs


def train_test_split(input_data: dict, test_size=0.3) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    X_train, X_test = {}, {}

    # X data split, y = person
    for person, imgs in input_data.items():
        X_train[person], X_test[person] = imgs[:7], imgs[7:]

    y_train = np.array([int(person)-1 for person in X_train.keys()])
    y_test = np.array([int(person)-1 for person in X_test.keys()])
    X_train = np.array([np.array([np.array(img) for img in person]) for person in X_train.values()])
    X_test = np.array([np.array([np.array(img) for img in person]) for person in X_test.values()])

    # Reshape (100, 7) -> (700, 1)
    y_train = torch.Tensor(np.array([label for label in y_train for _ in range(7)])).long()
    y_test = torch.Tensor(np.array([label for label in y_train for _ in range(3)])).long()
    X_train = torch.Tensor(np.array([img for person in X_train for img in person])).float()
    X_test = torch.Tensor(np.array([img for person in X_test for img in person])).float()

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


def train_squeezenet(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    model = model.to(device)
    
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
            
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss}")
    
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
    return f1_score(ys, predictions)



if __name__ == '__main__':
    
    mode = 'test'
    mode = 'preprocess'
    mode = 'train'
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    
    
    if mode == 'preprocess':
        # Read in input data
        input_data = read_raw()
        
        # Resize input data
        input_data = resize_input(input_data, tgt_size=224)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(input_data)
    
        # plt.imshow(X_train[0].permute(1,2,0), cmap='gray')
        # plt.show()
        # pass
    
        # Create train data set
        train_dataset = EarDataset(X_train, y_train)
        test_dataset = EarDataset(X_test, y_test)
        
        torch.save(train_dataset, "data/train_dataset.pt")
        torch.save(test_dataset, "data/test_dataset.pt")
    
    if mode == 'train':
        train_dataset = torch.load("data/train_dataset.pt")
        
        # Create train data loader
        train_dataloader = get_train_data(train_dataset)
        
        # data_iter = iter(train_dataloader)
        # sample_img, sample_label = next(data_iter)
        
        # print(type(sample_img[0][0]), type(sample_label[0]))
        # plt.imshow(sample_img[0])
        # plt.show()    
        
        # Get model and modify classifier
        model = get_squeezenet()
        n_classes = 100 
        model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))

        # Train model
        model = train_squeezenet(model)

        torch.save(model, "models/squeezenet.pt")
        
    if mode == 'test':
        
        test_dataloader = get_test_data(torch.load("data/test_dataset.pt"))
        train_dataloader = get_train_data(torch.load("data/train_dataset.pt"))

        
        # data_iter = iter(test_dataloader)
        # sample_img, sample_label = next(data_iter)
        # print(sample_label)
        # plt.imshow(sample_img[0].permute(1, 2, 0))
        # plt.show()  
        

        model = torch.load("models/squeezenet.pt")        
        
        score = test_squeezenet(model)
        print(score)
        
        

import torch
from torch import nn
import torchvision
from torchvision import transforms

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings

from ear_dataset import EarDataset


def get_train_data(train_dataset, batch_size=32):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def get_test_data(test_dataset, batch_size=32):
    trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def get_model(name="resnet"):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)  # test 50
    
    if name == "squeezenet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=False)

    elif name == "densenet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=False)

    return model


def train(model):
    torchvision.disable_beta_transforms_warning()
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 80
    model = model.to(device)
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for img_batch, label_batch in train_dataloader:
            torchvision.disable_beta_transforms_warning()
            
            
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

def get_mean_std(dataset):
    resize = transforms.Resize([224, 224])
    mr, mb, mg = [], [], []
    stdr, stdb, stdg = [], [], []
    cols = [[],[],[]]
    for img in dataset:
        img = img.transpose(2, 0, 1)
        img = resize(torch.Tensor(img))
        img = img / 255.0

        img_colors = [np.array(img[:, :, 0]).reshape(1, -1),
                        np.array(img[:,:,1]).reshape(1, -1),
                        np.array(img[:,:,2]).reshape(1, -1)]
        cols[0].extend(img_colors[0])
        cols[1].extend(img_colors[1])
        cols[2].extend(img_colors[2])
        
        mr.append(np.mean(img_colors[0]))
        mg.append(np.mean(img_colors[1]))
        mb.append(np.mean(img_colors[2]))
        
        stdr.append(np.std(img_colors[0]))
        stdg.append(np.std(img_colors[1]))
        stdb.append(np.std(img_colors[2]))
            
            # img_colors = img.transpose(2, 0, 1).reshape(3, -1)
            # colors = np.concatenate((colors, img_colors), axis=1)

    means = [np.mean(mr), np.mean(mg), np.mean(mb)]
    stds = [np.std(stdr), np.std(stdg), np.std(stdb)]

    means = [np.mean(cols[0]), np.mean(cols[1]), np.mean(cols[2])]
    stds = [np.std(cols[0]), np.std(cols[1]), np.std(cols[2])]

    return means, stds



if __name__ == '__main__':
    
    torchvision.disable_beta_transforms_warning()
    warnings.filterwarnings("ignore")
    
    torch.cuda.empty_cache()
    
    mode = 'test'
    mode = 'other'
    mode = 'train'
    mode = 'preprocess'
    
    if mode == "preprocess" or mode == "other":
        from preprocess import resize_input, train_test_split, read_raw
        
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    
    model_name = "squeezenet"
    model_name = "densenet"
    model_name = "resnet"
    print(model_name)
    
    
    # Read in input data
    input_data = read_raw()
    
    if mode == 'preprocess':
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(input_data)

        # Preprocess data
        X_train = resize_input(X_train, tgt_size=224, mode="train")       
        X_test = resize_input(X_test, tgt_size=224, mode="test")   
        
        fig, ax = plt.subplots(3,3)
        for i in range(1, 10):    
            print(X_train[i].shape)
            img = X_train[i+10*i].permute(1,2,0)
            ax[i//3 - 1][i%3].imshow(img)
        plt.show()
        print(input_data["001"][0].shape)
        
    
        # Create train data set
        train_dataset = EarDataset(X_train, y_train)
        test_dataset = EarDataset(X_test, y_test)
        
        torch.save(train_dataset, "data/train_dataset.pt")
        torch.save(test_dataset, "data/test_dataset.pt")
    
    if mode == 'train':
        train_dataset = torch.load("data/train_dataset.pt")
        
        # Create train data loader
        train_dataloader = get_train_data(train_dataset)   
        
        # Get model and modify classifier
        model = get_model(model_name)
        
        n_classes = y_train.max() + 1
        print(n_classes) 
        
        # Change structure of classifier
        if model_name == "squeezenet":
            model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
        elif model_name == "resnet":
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_classes)
        elif model_name == "densenet":
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, n_classes)
            
        # Train model
        model = train(model)

        torch.save(model, f"models/{model_name}.pt")
        
    if mode == 'test':
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
            fig, ax = plt.subplots(1,2)
            
            ax[0].imshow(test_imgs[i].permute(1, 2, 0))
            ax[0].set_title(f"Truth: {truth}")
            
            ax[1].imshow(input_data["{:03d}".format(pred)][0])
            ax[1].set_title(f"Pred: {pred}")

            plt.show()
        
    if mode == 'other':    
        
        # X_train, X_test, y_train, y_test = train_test_split(input_data)
        
        train_dataset = torch.load("data/train_dataset.pt")
        test_dataset = torch.load("data/test_dataset.pt")
        
        test_dataloader = get_test_data(test_dataset)
        train_dataloader = get_train_data(train_dataset)
        
        train_iter = iter(train_dataloader)
        test_iter = iter(test_dataloader)
        
        train_batch = next(train_iter)
        test_batch = next(test_iter)
        
        train_imgs, train_labels = train_batch
        test_imgs, test_labels = test_batch
        
        for i in range(10):
            fig, ax = plt.subplots(1,2)
            
            ax[0].imshow(train_imgs[i].permute(1,2,0))
            ax[0].set_title(f"Train: {train_labels[i]}")
            
            ax[1].imshow(test_imgs[i].permute(1,2,0))
            ax[1].set_title(f"Test: {test_labels[i]}")
            
            plt.show()
        # mean, std = get_mean_std(X_train)
        # print(mean, std)
        
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
        

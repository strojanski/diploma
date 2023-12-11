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

from ear_dataset import EarDataset, EarTriplet
from triplet import TripletLoss

parser = argparse.ArgumentParser(description="Person recognition on ear dataset.")
parser.add_argument(
    "--mode", type=str, default="train", help="Options: train, test, preprocess, other"
)
parser.add_argument(
    "--bs", type=int, default=64, help="Batch size"
)

parser.add_argument(
    "--id", type=str, default="1",
)

parser.add_argument(
    "--iter", type=int, default=8,
)



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
    model = models.googlenet(pretrained=True)
    return model


def train(model):
    torchvision.disable_beta_transforms_warning()

    criterion = TripletLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1, verbose=True
    )
    epochs = 200
    model = model.to(device)

    loss_history = []

    for epoch in range(epochs):

        epoch_loss = 0
        for img_batch, label_batch in train_dataloader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(img_batch)
            loss_ = criterion(output, label_batch)

            # Backward pass
            loss_.backward()

            optimizer.step()

            epoch_loss += loss_.item()
            loss_history.append(loss_.item())

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model, f"models/{model_name}_{id}_{epoch}_{iter_}.pt")
            np.savetxt(f"data/loss_/loss_history_{model_name}_{id}_{iter_}.txt", loss_history, fmt="%f", delimiter=",")

        scheduler.step()

        print(f"Epoch: {epoch+1}/{epochs}, loss_: {epoch_loss}")

    np.savetxt(f"data/loss_/loss_history_{model_name}_{id}_{epoch}_{batch_size}_{iter_}.txt", loss_history, fmt="%f", delimiter=",")
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
        # print(f"Class {label}: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
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


def split_triplets(X, y):
    # Constraint: anchor and positive must be the same person
    X, y = np.array(X), np.array(y)
    print("Classes: ", len(np.unique(y)))

    label_to_indices = {label: np.where(y == label)[0] for label in np.unique(y)}
    
    for key in label_to_indices.keys():
        np.random.shuffle(label_to_indices[key])

    an, pos, neg = [], [], []

    negs = np.roll(np.array(list(label_to_indices.keys())), shift=1)
    print(negs, "\n")

    for i, label in enumerate(label_to_indices.keys()):
        subarray_size = len(label_to_indices[label]) // 3 + 1
        print(subarray_size)

        # anchor and positive have label = label and negative is random
        an.extend(label_to_indices[label][:subarray_size])
        pos.extend(label_to_indices[label][subarray_size:2 * subarray_size])
        neg.extend(label_to_indices[negs[i]][:subarray_size])

        # Split the shuffled indices into three arrays with the desired constraints
        # array1_indices = np.concatenate([indices[:subarray_size] for indices in label_to_indices.values()])
        # array2_indices = np.concatenate([indices[subarray_size:2 * subarray_size] for indices in label_to_indices.values()])
        # array3_indices = np.concatenate([indices[2 * subarray_size:] for indices in label_to_indices.values()])
        
        # Shuffle the resulting indices to mix labels
        # np.random.shuffle(array1_indices)
        # np.random.shuffle(array2_indices)
        # np.random.shuffle(array3_indices)

    print(an, pos, neg)

    # Use the resulting arrays of indices to get the split arrays
    anchor_data, anchor_labels = X[an], y[an]
    positive_data, positive_labels = X[pos], y[pos]
    negative_data, negative_labels = X[neg], y[neg]
        
    print(anchor_data.shape, positive_data.shape, negative_data.shape)

            
    return (anchor_data, anchor_labels), (positive_data, positive_labels), (negative_data, negative_labels)



if __name__ == "__main__":
    torchvision.disable_beta_transforms_warning()
    warnings.filterwarnings("ignore")

    torch.cuda.empty_cache()

    mode = parser.parse_args().mode
    
    print(mode)
    batch_size = parser.parse_args().bs
    id = parser.parse_args().id
    iter_ = parser.parse_args().iter
        
    from preprocess import read_raw, resize_input, train_test_split

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(device)
    
    model_name = "googlenet"

    print(model_name)

    # Read in input data
    input_data = None

    if mode == "preprocess":
        input_data = read_raw()
        print("Read input data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(input_data)

        # Preprocess data
        X_train = resize_input(X_train, tgt_size=64, mode="train")
        X_test = resize_input(X_test, tgt_size=64, mode="test")

        if False:
            fig, ax = plt.subplots(3, 3)
            for i in range(1, 10):
                print(X_train[i].shape)
                img = X_train[i + 10 * i].permute(1, 2, 0)
                ax[i // 3 - 1][i % 3].imshow(img)
            plt.show()
            print(input_data["0001"][0].shape)


        # Split to anchor, positive and negative
        train_anchor, train_positive, train_negative = split_triplets(X_train, y_train)
        test_anchor, test_positive, test_negative = split_triplets(X_test, y_test)

        # Create train data set
        train_dataset = EarTriplet(train_anchor[0], train_anchor[1], train_positive[0], train_positive[1], train_negative[0], train_negative[1])
        test_dataset = EarTriplet(test_anchor[0], test_anchor[1], test_positive[0], test_positive[1], test_negative[0], test_negative[1])

        torch.save(train_dataset, f"data/train_dataset_{id}.pt")
        torch.save(test_dataset, f"data/test_dataset_{id}.pt")

    if mode == "train":
        train_dataset = torch.load(f"data/train_dataset_64_all.pt")
        train_dataset.labels_to_long()

        # Create train data loader
        train_dataloader = get_train_data(train_dataset, batch_size=batch_size)

        m = "new"
        m = "old"

        model = None
        if m == "new":
            # Get model and modify classifier
            model = get_model(f"{model_name}_{id}")
            # model.fc = nn.Identity()

            model = list(model.children())[:-1]

            print(
                f"{len(set(train_dataset.labels)), min(train_dataset.labels), max(train_dataset.labels)}"
            )

            n_classes = train_dataset.get_n_classes()
            print(f"Number of classes: {n_classes}")
            print(f"Number of images: {len(train_dataset.data)}")

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
            # model = torch.load(f"models/{model_name}_{id}.pt")
            model = torch.load(f"models/{model_name}_{id}_final_11.pt")
            model.train()

        # Train model
        model = train(model)

        torch.save(model, f"models/{model_name}_{id}_final_{iter_}.pt")

    if mode == "test":
        test_dataset = torch.load(f"data/test_dataset_{id}.pt")
        train_dataset = torch.load(f"data/train_dataset_{id}.pt")
        
        ad, al, pd, pl, nd, nl = train_dataset.get_data(0)
        print("Testing...")
        print(al, pl, nl)      

        print(len(test_dataset), len(train_dataset))

        for i in range(len(test_dataset)):
            print(test_dataset.anchor_labels[i], test_dataset.positive_labels[i], test_dataset.negative_labels[i])
        
        
        test_dataloader = get_test_data(test_dataset, batch_size=batch_size)
        train_dataloader = get_train_data(train_dataset, batch_size=batch_size)

        test_imgs, test_labels = next(iter(test_dataloader))

        pass

        model = torch.load(f"models/{model_name}.pt")
        model = torch.load(f"models/{model_name}_{id}_180_{iter_}.pt")

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

    if mode == "analysis":
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from torchvision.models import resnet50

        model = torch.load(f"models/{model_name}_{id}_final_{iter_}.pt").to("cpu")

        target_layers = [model.fc]#[getattr(model, "inception4a")]
        
        test_dataset = torch.load(f"data/test_dataset_240.pt")
        test_dataloader = get_test_data(test_dataset, batch_size=batch_size)
        
        batch = next(iter(test_dataloader))
        img_batch, label_batch = batch

        rgb_img = img_batch[0]#.permute(1, 2, 0)
        rgb_img = torch.unsqueeze(rgb_img, 0)
        input_tensor = rgb_img # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!
        print(input_tensor.shape)

        from pytorch_grad_cam import DeepFeatureFactorization
        from pytorch_grad_cam.utils.image import show_factorization_on_image
        dff = DeepFeatureFactorization(model=model, target_layer=model.fc, computation_on_concepts=model.fc)
        print(dff)
        concepts, batch_explanations, concept_scores = dff(input_tensor, n_components=1024)
        visualization = show_factorization_on_image(rgb_img, 
                                                    batch_explanations[0],
                                                    image_weight=0.3)

        print(rgb_img.shape)
        

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        targets = [ClassifierOutputTarget(281)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        plt.imshow(rgb_img)
        plt.show()

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if mode == "other":
        # X_train, X_test, y_train, y_test = train_test_split(input_data)

        train_dataset = torch.load(f"data/train_dataset.pt")
        test_dataset = torch.load(f"data/test_dataset.pt")
        print(f"Number of images: {len(train_dataset.data)}")

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
       

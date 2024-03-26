import torch, sys, datetime, random
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import torch.nn as nn
import namegenerator


RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + " " + namegenerator.gen()
random.seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.append("../data")
from uerc_dataset import UERCDataset

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == 'cpu':
    print('CPU is used')
    exit()

DATA_BASE_PATH = '../data/public'
BATCH_SIZE = 512
PRELOAD_MODEL = None

NUM_EPOCHS = 100

data_train = UERCDataset('train', DATA_BASE_PATH, '../data/public_annotations.csv', '../runs/public_image_list.csv', data_splits_csv='../runs/train_val_test_splits.csv')
data_val = UERCDataset('val', DATA_BASE_PATH, '../data/public_annotations.csv', '../runs/public_image_list.csv', data_splits_csv='../runs/train_val_test_splits.csv')

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False)

def save_best_model(model, accuracy, best_accuracy, file_path):
    if accuracy >= best_accuracy:
        torch.save(model.state_dict(), file_path)
        best_accuracy = accuracy
        print('Best model (' + str(best_accuracy) + ') saved')
    return best_accuracy

# initialize best_accuracy as 0 or some low value
best_accuracy = 0.0

# Define the model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, )
model.train()

for name, module in model.named_children():
    if isinstance(module, nn.Conv2d):
        module = nn.Sequential(module, nn.Dropout2d(p=0.2, inplace=True))
        setattr(model, name, module)
RUN_ID += '-resnet18-dropout0.2'

# Freeze the model parameters
for param in model.parameters():
    param.requiresGrad = False

# Replace the last fully connected layer
num_ftrs = model.fc.in_features
classes_count = data_train.num_of_classes['train']
model.fc = torch.nn.Linear(num_ftrs, classes_count)

# if model exists, load it:
if PRELOAD_MODEL is not None:
    p = os.path.join('models', PRELOAD_MODEL)
    if os.path.exists(p):
        print('Loading model from: ' + p)
        model.load_state_dict(torch.load(p))

# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001, betas=(0.8, 0.999), eps=1e-08)

print('Starting training, run id: ' + RUN_ID)

# Start training
for epoch in range(NUM_EPOCHS):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, NUM_EPOCHS, loss.item(), accuracy))
    best_accuracy = save_best_model(model, accuracy, best_accuracy, RUN_ID + '.pt')

print('Finished training, run id: ' + RUN_ID)
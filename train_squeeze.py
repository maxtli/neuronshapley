# train a squeezenet on CelebA gender classification; we use torchvision and train efficiently on a single GPU.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# set up the data (download)
transform = transforms.Compose(
    [
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# dataset is at data/celeba, with images at data/celeba/img_align_celeba/img_align_celeba/*.jpg
trainset = torchvision.datasets.ImageFolder(root="data/celeba", transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)


testset = torchvision.datasets.ImageFolder(root="data/celeba", transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

# set up the model
model = torchvision.models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2
model = model.to(device)

# set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the model (use tqdm)
from tqdm import tqdm
import time


def train(model, trainloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_acc.double() / len(trainloader.dataset)
        print(
            "Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(
                epoch + 1, epoch_loss, epoch_acc
            )
        )


# test the model
def test(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(testloader.dataset)
    epoch_acc = running_acc.double() / len(testloader.dataset)
    print("Test Loss: {:.4f}, Test Acc: {:.4f}".format(epoch_loss, epoch_acc))


# train the model
train(model, trainloader, criterion, optimizer, device, epochs=1)

# test the model
test(model, testloader, criterion, device)

# save the model
torch.save(model.state_dict(), "squeezenet.pth")

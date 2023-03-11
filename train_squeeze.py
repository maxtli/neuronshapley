# train a squeezenet on CelebA gender classification; we use torchvision and train efficiently on a single GPU.
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import PIL
from tqdm import tqdm
import os

# set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%

# treat the image_id column as the id
df = pd.read_csv("data/celeba/list_attr_celeba.csv", index_col="image_id")
genders = df["Male"]

# %%


class CelebDataset(Dataset):
    def __init__(self, img_dir, transform, target_transform=None):
        self.img_dir = img_dir  # where all the images are
        self.img_labels = list(sorted(os.listdir(img_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = PIL.Image.open(img_path)
        label = int((genders[self.img_labels[idx]] + 1) / 2)  # convert from -1,1 to 0,1
        image = self.transform(image)
        return image, label


# %%

transform = transforms.Compose(
    [
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_set = CelebDataset(img_dir="data/celeba/train", transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)

val_set = CelebDataset(img_dir="data/celeba/val", transform=transform)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

# %%

# display an image with PIL
idx = 61
plt.imshow(val_set[idx][0].permute(1, 2, 0))
print("Male" if val_set[idx][1] > 0 else "Female")

# %%

# set up the model
model = torchvision.models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2
model = model.to(device)

# train the model (use tqdm)


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
            running_acc += torch.sum(preds == labels.data) / len(labels)
            if i % 1000 == 999:
                tqdm.write(
                    "Epoch: {}, Iteration: {}, Loss: {:.4f}, Acc: {:.4f}".format(
                        epoch + 1, i + 1, running_loss / 1000, running_acc / 1000
                    )
                )
                running_loss = 0.0
                running_acc = 0.0


# test the model
def val(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels.data) / len(labels)
    epoch_loss = running_loss / len(valloader)
    epoch_acc = running_acc / len(valloader)
    tqdm.write("Val Loss: {:.4f}, Val Acc: {:.4f}".format(epoch_loss, epoch_acc))


# %%

# set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the model
train(model, train_loader, criterion, optimizer, device, epochs=1)

# %%

# test the model
val(model, val_loader, criterion, device)

# save the model
torch.save(model.state_dict(), "squeezenet.pth")

# %%
# load from ./squeezenet.pth
model_loaded = torchvision.models.squeezenet1_1(pretrained=False)
model_loaded.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model_loaded.num_classes = 2
model_loaded.load_state_dict(torch.load("squeezenet.pth"))
model_loaded = model_loaded.to(device)

# %%

val(model_loaded, val_loader, criterion, device)

# %%
# file,age,gender,race,service_test

df = pd.read_csv("data/fairface/fairface_label_val.csv", index_col="file")

ages = (
    (df["age"] != "3-9")
    * (df["age"] != "10-19")
    * (df["age"] != "60-69")
    * (df["age"] != "more than 70")
)
white_men = df.loc[(df["race"] == "White") * (df["gender"] == "Male") * ages][:500]
white_women = df.loc[(df["race"] == "White") * (df["gender"] == "Female") * ages][:500]
black_men = df.loc[(df["race"] == "Black") * (df["gender"] == "Male") * ages][:500]
black_women = df.loc[(df["race"] == "Black") * (df["gender"] == "Female") * ages][:500]

# %%


class FairfaceDataset(Dataset):
    def __init__(self, img_dir, transform, target_df):
        self.img_dir = img_dir
        self.img_labels = target_df.index  # the filenames
        self.transform = transform
        self.target = int(target_df["gender"][0] == "Male")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = PIL.Image.open(img_path)
        image = self.transform(image)
        return image, self.target


white_men_ds = FairfaceDataset("data/fairface", transform, white_men)
white_women_ds = FairfaceDataset("data/fairface", transform, white_women)
black_men_ds = FairfaceDataset("data/fairface", transform, black_men)
black_women_ds = FairfaceDataset("data/fairface", transform, black_women)

# %%
img = white_men_ds[0][0].permute(1, 2, 0)
denormalized_img = img * torch.tensor([0.229, 0.224, 0.225]).view(
    1, 1, 3
) + torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
plt.imshow(denormalized_img)

# %%

# evaluate performance on the four datasets
def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels).item() / len(labels)
    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)
    tqdm.write("Val Loss: {:.4f}, Val Acc: {:.4f}".format(epoch_loss, epoch_acc))


# %%

evaluate(model_loaded, white_men_ds)
print("--ABOVE IS WHITE MEN--\n")
evaluate(model_loaded, white_women_ds)
print("--ABOVE IS WHITE WOMEN--\n")
evaluate(model_loaded, black_men_ds)
print("--ABOVE IS BLACK MEN--\n")
evaluate(model_loaded, black_women_ds)
print("--ABOVE IS BLACK WOMEN--\n")

# %%


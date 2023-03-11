# train a squeezenet on CelebA gender classification.

# %%

from datasets import get_celeb_data_loaders, get_gender_dataloaders
from model import get_squeezenet, epoch

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# %%

# whether or not to save weights after training
save_run = False

# get celebA data
train_loader, val_loader = get_celeb_data_loaders(batch_size=32, num_workers=2)

# get model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device)
model = get_squeezenet(weights_path=None, device=device)  # "squeezenet.pth"

# %%

# set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 2

print("initial validation...")
epoch(model, val_loader, criterion, None, device)

print("training...")
for i in range(epochs):
    epoch(model, train_loader, criterion, optimizer, device)

print("final validation...")
epoch(model, val_loader, criterion, None, device)

# save the model
if save_run:
    torch.save(model.state_dict(), "squeezenet.pth")

# %%

print("\n\nEVAL ON FAIRFACE\n\n")
white_men, white_women, black_men, black_women = get_gender_dataloaders()

epoch(model, white_men, criterion, None, device)
print("--ABOVE IS WHITE MEN--\n")
epoch(model, white_women, criterion, None, device)
print("--ABOVE IS WHITE WOMEN--\n")
epoch(model, black_men, criterion, None, device)
print("--ABOVE IS BLACK MEN--\n")
epoch(model, black_women, criterion, None, device)
print("--ABOVE IS BLACK WOMEN--\n")

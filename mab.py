# %%
import torchvision
import torch as t
import torch.nn.functional as f
import numpy as np
import math
from torch import nn
from functools import partial
from torch.utils.data import TensorDataset
import random
from tqdm import tqdm
import pickle
import time
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL
from multiprocessing import Pool


# %%

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


transform = transforms.Compose(
    [
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

val_set = CelebDataset(img_dir="data/celeba/val", transform=transform)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

# %%


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

# combine all datasets into one dataloader
combined_dataset = t.utils.data.ConcatDataset(
    [white_men_ds, white_women_ds, black_men_ds, black_women_ds]
)
combined_dataloader = DataLoader(
    combined_dataset, batch_size=128, shuffle=True, num_workers=2
)

# # batch is entire dataset
# white_men_dataloader = DataLoader(
#     white_men_ds, batch_size=500, shuffle=True, num_workers=2
# )
# white_women_dataloader = DataLoader(
#     white_women_ds, batch_size=500, shuffle=True, num_workers=2
# )
# black_men_dataloader = DataLoader(
#     black_men_ds, batch_size=500, shuffle=True, num_workers=2
# )
# black_women_dataloader = DataLoader(
#     black_women_ds, batch_size=500, shuffle=True, num_workers=2
# )

# %%
model = torchvision.models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
model.load_state_dict(t.load("squeezenet.pth"))
model = model.to(device)
model.eval()

# %%

convs = list(filter(lambda x: isinstance(x, nn.Conv2d), list(model.modules())))

filt_len = sum([c.out_channels for c in convs])

ablations = np.zeros(filt_len)

# %%

# SKIP if not storing means

mean_dict = dict()
# store mean of every filter, calculated from val_loader CelebA data
def mean_hook(module, input, output):
    means = t.mean(output.relu(), dim=0, keepdim=True)
    # add value to mean_dict[module] if it exists, otherwise create it
    mean_dict[module] = mean_dict.get(module, 0) + means
    return output


def forward_pass_store_means(loader):
    handlers = []
    for conv in convs:
        handlers.append(conv.register_forward_hook(mean_hook))

    with t.no_grad():
        running_loss = running_acc = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += f.cross_entropy(output, target)
            pred = output.argmax(dim=1)
            running_acc += (pred == target).sum().item() / len(pred)

    acc = running_acc / len(loader)

    for h in handlers:
        h.remove()


forward_pass_store_means(val_loader)
for key in mean_dict:
    mean_dict[key] /= len(val_loader)

# %%


def mean_ablation_hook(module, input, output, ablate_mask):
    output[:, ablate_mask] = mean_dict[module][:, ablate_mask]
    return output


# ablations[i] is 1 if we want to ablate neuron i
def forward_pass(ablations, loader):
    start_idx = 0
    handlers = []
    for conv in convs:
        ablate_mask = ablations[start_idx : start_idx + conv.out_channels] == 1
        handlers.append(
            conv.register_forward_hook(
                partial(mean_ablation_hook, ablate_mask=ablate_mask)
            )
        )
        start_idx += conv.out_channels
    
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1)
    acc = (pred == target).sum().item() / len(pred)

    for h in handlers:
        h.remove()

    return acc


# %%

truncate_threshold = 0.52
epsilon = 0.001

relevant_neurons = set(range(filt_len))

# data_loaders = [
#     white_men_dataloader,
#     white_women_dataloader,
#     black_men_dataloader,
#     black_women_dataloader,
# ]

all_neuron_acc = forward_pass(np.zeros(filt_len), combined_dataloader)
# all_neuron_acc = sum(forward_pass(ablate_mask, data_loaders)) / len(data_loaders)

# start = time.time()
# forward_pass_normal(combined_dataloader)
# print("all neuron acc", all_neuron_acc)
# print("time", time.time() - start)


# %%
shapley_values = np.zeros(filt_len)
variances = np.zeros(filt_len)

# # confidence bounds are symmetric: the intervals are [shapley_values[i] +- cb[i]]
cb = np.zeros(filt_len)
samples = np.zeros(filt_len)

max_loss = 1


def update_once(delta, k):
    seq = np.random.permutation(np.arange(filt_len))
    ablate_mask = np.zeros(filt_len)

    global relevant_neurons

    prev_loss = all_neuron_acc

    def update_loss():
        # start = time.time()
        out = (
            forward_pass(ablate_mask, combined_dataloader)
            if prev_loss >= truncate_threshold
            else prev_loss
        )
        # print("update loss time", time.time() - start)
        return out

    # remove neurons by adding them to the ablate mask
    for i, neuron in enumerate(tqdm(seq)):
        ablate_mask[neuron] = 1
        if neuron not in relevant_neurons:
            if i < len(seq) - 1 and seq[i + 1] in relevant_neurons:
                prev_loss = update_loss()
            continue
        loss = update_loss()
        differential = loss - prev_loss

        # prev_loss is the loss with neuron i included. loss is the loss with neuron i excluded
        samples[neuron] += 1
        value_update = (differential - shapley_values[neuron]) / samples[neuron]

        shapley_values[neuron] += value_update

        variances[neuron] = (
            (samples[neuron] - 1) * (variances[neuron] + value_update ** 2)
            + (differential - shapley_values[neuron]) ** 2
        ) / samples[neuron]

        if i % 30 == 0:
            print("prev loss", prev_loss)

        # I think ghorbani forgets to do a union bound here, twice.
        # First, the pulls are not independent (previous pulls influence which future pulls to do).
        # Second, you want to guarantee that wp 1-delta, ALL confidence bounds are valid, not just 1.
        # original code:
        #     cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / delta) / counts[counts > 1]) +\
        # 7/3 * R * np.log(2 / delta) / (counts[counts > 1] - 1)
        cb[neuron] = (
            math.sqrt(2 * variances[neuron] * math.log(2 / delta) / samples[neuron])
            + 7 / 3 * math.log(2 / delta) / (samples[neuron] - 1)
            if samples[neuron] > 1
            else max_loss
        )

        prev_loss = loss

    # I don't like the threshold approach, seems like the wrong thing to do. What really matters is that the confidence bounds don't overlap, this feels like it goes for longer than necessary. Also, why not just pull top-k?

    # this gets the kth largest shapley value
    threshold = np.partition(shapley_values, -k)[-k]

    # the relevant neurons are the ones to be pulled next
    # cb[relevant_neurons] = confidence_bounds(samples[relevant_neurons], variances[relevant_neurons], delta)
    relevant_neurons = set(
        [
            i
            for i in range(filt_len)
            if shapley_values[i] + cb[i] > threshold + epsilon
            and shapley_values[i] - cb[i] < threshold - epsilon
        ]
    )

    return len(relevant_neurons)


delta = 0.1
k = 100
i = 0

# %% 
for i in range(200):
    num_relevant = update_once(delta, k)
    if i % 5 == 0:
        pickle.dump("iterations.pkl", i)
        pickle.dump("shapley_values.pkl", shapley_values)
        pickle.dump("variances.pkl", variances)
        pickle.dump("cb.pkl", cb)
        pickle.dump("samples.pkl", samples)

    if num_relevant == 1:
        break

neurons = np.sort(np.partition(shapley_values, -k)[-k:])
print(neurons)

# # samples = total number of samples
# def moving_average(samples, prev_mean, x_n):
#     prev_mean[relevant_neurons]
#     return ((samples-1) * prev_mean + x_n) / samples

# def moving_variance(samples, prev_mean, prev_var, new_mean, x_n):
#     return ((samples-1) * (prev_var + np.square(new_mean - prev_mean)) + np.square(x_n - new_mean)) / samples

# def confidence_bounds(samples, variances, delta):
#     return np.sqrt(2 * variances * np.log(2 / delta) / samples) + 7/3 * np.log(2 / delta) / (samples-1)

# shapley_values += value_updates
# variances[relevant_neurons] = ((samples[relevant_neurons] - 1) * (variances[relevant_neurons] + np.square(value_updates[relevant_neurons])) + np.square(differential[relevant_neurons] - shapley_values[relevant_neurons])) / samples[relevant_neurons]


# %%

with Pool(3) as p:
    print(p.starmap(forward_pass, [(np.zeros(filt_len), combined_dataset), (np.zeros(filt_len), combined_dataset), (np.zeros(filt_len), combined_dataset)]))
# %%

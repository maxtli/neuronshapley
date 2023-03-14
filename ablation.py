# %%
from datasets import get_combined_gender_loader, get_celeb_data_loaders, get_gender_dataloaders
from model import get_squeezenet, epoch

import torch as t
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from tqdm import tqdm
import pickle

# %%

# configure model and data
device = "cuda" if t.cuda.is_available() else "cpu"
model = get_squeezenet(weights_path="squeezenet.pth", device=device)

celeba_train_loader, celeba_val_loader = get_celeb_data_loaders(
    batch_size=128, num_workers=2
)
genders_loader = get_combined_gender_loader(batch_size=128, num_workers=2)

# %%

# get all conv layers (where we'll be ablating)
convs = list(filter(lambda x: isinstance(x, nn.Conv2d), list(model.modules())))

# get total number of conv filters
filt_len = sum([c.out_channels for c in convs])

# %%

# store the order of convolutional modules so we can look up the means
conv_dict = dict()
for idx, c in enumerate(convs):
    conv_dict[c] = idx

# get means of every filter, calculated from val_loader CelebA data
def mean_hook(module, input, output, conv_means):
    means = t.mean(output.relu(), dim=0, keepdim=True)
    # print(means.shape)
    # print(conv_means[conv_dict[module]].shape)
    # print("----")
    conv_means[conv_dict[module]] = means
    return output

# calculating the means for every filter takes 16 seconds, so we store it in a file
def forward_pass_store_means(loader):
    conv_means = [t.zeros((1,)) for c in convs]

    handlers = []
    for conv in convs:
        handlers.append(conv.register_forward_hook(partial(mean_hook, conv_means=conv_means)))

    epoch(model, loader, nn.CrossEntropyLoss(), None, device)

    for h in handlers:
        h.remove()

    for i, _ in enumerate(conv_means):
        conv_means[i] /= len(celeba_val_loader)

    with open("means.pkl", "wb") as f:
        pickle.dump(conv_means, f)

# load the means from saved file
def load_conv_means():
    with open("means.pkl", "rb") as f:
        return pickle.load(f)

# %%

# hook for ablating filters during forward pass according to `ablate_mask`
def mean_ablation_hook(module, input, output, ablations, conv_means):
    output[:, ablations] = conv_means[conv_dict[module]][:, ablations]
    return output

# hook for zero ablating filters during forward pass according to `ablate_mask`
def zero_ablation_hook(module, input, output, ablations):
    output[:, ablations] = 0
    return output

# forward pass with ablation for *a single batch* from loader
# 1 means ablate the filter. 0 means don't ablate
def forward_pass(ablate_mask, conv_means, loader, full_data=True, zero_ablate=False):
    start_idx = 0
    handlers = []
    for conv in convs:
        ablations = ablate_mask[start_idx : start_idx + conv.out_channels] == 1
        handlers.append(
            conv.register_forward_hook(
                partial(mean_ablation_hook, ablations=ablations, conv_means=conv_means) if not zero_ablate else partial(zero_ablation_hook, ablations=ablations)
            )
        )
        start_idx += conv.out_channels

    if full_data:
        acc = epoch(model, loader, nn.CrossEntropyLoss(), None, device, False) 
    else:
        with t.no_grad():
            data, target = next(iter(loader))
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            acc = (pred == target).sum().item() / len(pred)

    for h in handlers:
        h.remove()

    return acc
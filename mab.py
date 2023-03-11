# %%
from datasets import get_combined_gender_loader, get_celeb_data_loaders
from model import get_squeezenet, epoch

import torch as t
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import math
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

# form ablation array, where 0 -> don't ablate, and 1 -> ablate.
ablations = np.zeros(filt_len)

# %%

# get means of every filter, calculated from val_loader CelebA data
mean_dict = dict()


def mean_hook(module, input, output):
    means = t.mean(output.relu(), dim=0, keepdim=True)
    # add value to mean_dict[module] if it exists, otherwise create it
    mean_dict[module] = mean_dict.get(module, 0) + means
    return output


def forward_pass_store_means(loader):
    handlers = []
    for conv in convs:
        handlers.append(conv.register_forward_hook(mean_hook))

    epoch(model, loader, nn.CrossEntropyLoss(), None, device)

    for h in handlers:
        h.remove()


forward_pass_store_means(celeba_val_loader)
for key in mean_dict:
    mean_dict[key] /= len(celeba_val_loader)

# %%

# hook for ablating filters during forward pass according to `ablate_mask`
def mean_ablation_hook(module, input, output, ablate_mask):
    output[:, ablate_mask] = mean_dict[module][:, ablate_mask]
    return output


# forward pass with ablation for *a single batch* from loader
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

    with t.no_grad():
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
all_neuron_acc = forward_pass(np.zeros(filt_len), combined_dataloader)

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
# for i in range(200):
#     num_relevant = update_once(delta, k)
#     if i % 5 == 0:
#         with open("iterations.pkl", "wb") as pickle_file:
#             pickle.dump(i, pickle_file)
#         with open("shapley_values.pkl", "wb") as pickle_file:
#             pickle.dump(shapley_values, pickle_file)
#         with open("variances.pkl", "wb") as pickle_file:
#             pickle.dump(variances, pickle_file)
#         with open("cb.pkl", "wb") as pickle_file:
#             pickle.dump(cb, pickle_file)
#         with open("samples.pkl", "wb") as pickle_file:
#             pickle.dump(samples, pickle_file)

#     if num_relevant == 1:
#         break

# neurons = np.sort(np.partition(shapley_values, -k)[-k:])
# print(neurons)

# %%

# load in shapeley values
with open("shapley_values.pkl", "rb") as pickle_file:
    shapley_values = pickle.load(pickle_file)

shapley_values.sort()
print(sum(shapley_values))
print(sum(shapley_values[-100:]))

# %%

# batch is entire dataset
white_men_dataloader = DataLoader(
    white_men_ds, batch_size=500, shuffle=True, num_workers=2
)
white_women_dataloader = DataLoader(
    white_women_ds, batch_size=500, shuffle=True, num_workers=2
)
black_men_dataloader = DataLoader(
    black_men_ds, batch_size=500, shuffle=True, num_workers=2
)
black_women_dataloader = DataLoader(
    black_women_ds, batch_size=500, shuffle=True, num_workers=2
)

with open("shapley_values.pkl", "rb") as pickle_file:
    shapley_values = pickle.load(pickle_file)

neurons_sorted = np.argsort(shapley_values)  # ascending order


def check_score(ablate_mask):
    wm_acc = forward_pass(ablate_mask, white_men_dataloader)
    print("White men", wm_acc)
    ww_acc = forward_pass(ablate_mask, white_women_dataloader)
    print("White women", ww_acc)
    bm_acc = forward_pass(ablate_mask, black_men_dataloader)
    print("Black men", bm_acc)
    bw_acc = forward_pass(ablate_mask, black_women_dataloader)
    print("Black women", bw_acc)
    overall_acc = sum([wm_acc, ww_acc, bm_ac, bw_acc]) / 4
    print("overall", overall_acc)
    print(len(celeba_val_loader))
    celeba_acc = forward_pass(
        ablate_mask, celeba_val_loader
    )  # TODO uses new batch every iteration, weird
    print("celeba", celeba_acc)

    return wm_acc, ww_acc, bm_acc, bw_acc, overall_acc, celeba_acc


ablate_mask = np.zeros(filt_len)
check_score(ablate_mask)

wm_accs, ww_accs, bm_accs, bw_accs, overall_accs, celeba_accs = [], [], [], [], [], []
for i in tqdm(range(30)):
    ablate_mask[neurons_sorted[-i - 1]] = 1
    wm_acc, ww_acc, bm_ac, bw_acc, overall_acc, celeba_acc = check_score(ablate_mask)
    wm_accs.append(wm_acc)
    ww_accs.append(ww_acc)
    bm_accs.append(bm_ac)
    bw_accs.append(bw_acc)
    overall_accs.append(overall_acc)
    celeba_accs.append(celeba_acc)

# %%

import matplotlib.pyplot as plt

plt.hist(shapley_values, bins=100)
plt.show()

plt.plot(celeba_accs, label="celeba")
plt.plot(wm_accs, label="white men")
plt.plot(ww_accs, label="white women")
plt.plot(bm_accs, label="black men")
plt.plot(bw_accs, label="black women")
plt.plot(overall_accs, label="overall")
plt.legend()
plt.xlabel("Number of filters ablated")
plt.ylabel("Test Accuracy (%)")


# %%

# load in iterations.pkl
with open("iterations.pkl", "rb") as pickle_file:
    iterations_pk = pickle.load(pickle_file)
print(iterations_pk)


# %%

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

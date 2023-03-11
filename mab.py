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

# %% 
model = torchvision.models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

model = model.to(device)


# %%

convs = filter(lambda x: isinstance(x, nn.Conv2d), list(model.modules()))

filt_len = sum([c.out_channels for c in convs])

ablations = np.zeros(filt_len)

# ablate_mask[i] is 1 if we want to ablate neuron i
def mean_ablation_hook(module, input, output, ablate_mask):
    means = t.mean(output.relu(), dim=0, keepdim=True)
    output[:, ablate_mask] = means[:, ablate_mask]
    return output

intrinsic_values = np.arange(filt_len)
# ablations[i] is 1 if we want to ablate neuron i
def forward_pass(ablations):
    return np.sum((1-ablations) * intrinsic_values) + np.random.normal()

    model.eval()
    model.to(device)
    start_idx = 0
    handlers = []
    for conv in convs:
        ablate_mask = ablations[start_idx:start_idx+conv.out_channels]
        handlers.append(conv.register_forward_hook(partial(mean_ablation_hook, ablate_mask=ablate_mask)))
        start_idx += conv.out_channels
    
    # do the forward pass
    # return the accuracy/loss
    
    # data, target = dataset.tensors
    # data, target = data.to(device), target.to(device)
    # with t.inference_mode():
    #     logits = model(data)
    #     test_loss = f.cross_entropy(logits, target).item()
    #     pred = logits.argmax(dim=1)  # (n, )
    #     correct = pred == target  # (n, )
    # acc = correct.sum().item() / len(dataset)
    # three_acc = (pred[target == 3] == 3).sum().item() / num_threes
    # non_three_acc = (acc - num_threes / len(dataset) * three_acc) / (1 - num_threes / len(dataset))
    # return TestResult(
    #     loss=test_loss,
    #     acc=acc,
    #     threes_acc=three_acc,
    #     non_threes_acc=non_three_acc,
    # )

# %%

truncate_threshold = 0.1

seq = np.random.permutation(np.arange(filt_len))
relevant_neurons = set(range(filt_len))

ablate_mask = np.zeros(filt_len)
all_neuron_loss = forward_pass(ablate_mask)
shapley_values = np.zeros(filt_len)
variances = np.zeros(filt_len)

# # confidence bounds are symmetric: the intervals are [shapley_values[i] +- cb[i]]
cb = np.zeros(filt_len)
samples = np.zeros(filt_len)

max_loss = 1    

def update_once(delta, k):

    prev_loss = all_neuron_loss

    def update_loss():
        return forward_pass(ablate_mask) if prev_loss >= truncate_threshold else prev_loss

    # remove neurons by adding them to the ablate mask
    for i, _ in enumerate(seq):
        ablate_mask[seq[i]] = 1
        if seq[i] not in relevant_neurons:
            if seq[i+1] in relevant_neurons:
                prev_loss = update_loss()
            continue
        loss = update_loss()
        differential = loss - prev_loss

        # prev_loss is the loss with neuron i included. loss is the loss with neuron i excluded
        samples[i] += 1
        value_update = (differential - shapley_values[i]) / samples[i]

        shapley_values[i] += value_update
        variances[i] = ((samples[i] - 1) * (variances[i] + value_update ** 2) + (differential - shapley_values[i]) ** 2) / samples[i]
        
        # I think ghorbani forgets to do a union bound here, twice.
        # First, the pulls are not independent (previous pulls influence which future pulls to do).
        # Second, you want to guarantee that wp 1-delta, ALL confidence bounds are valid, not just 1.
        # original code:
        #     cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / delta) / counts[counts > 1]) +\
        # 7/3 * R * np.log(2 / delta) / (counts[counts > 1] - 1)
        cb[i] = math.sqrt(2 * variances[i] * math.log(2 / delta) / samples[i]) + 7/3 * math.log(2 / delta) / (samples[i]-1) if samples[i] > 1 else max_loss

        prev_loss = loss
        
    # I don't like the threshold approach, seems like the wrong thing to do. What really matters is that the confidence bounds don't overlap, this feels like it goes for longer than necessary. Also, why not just pull top-k?

    # this gets the kth largest shapley value
    threshold = np.partition(shapley_values, -k)[-k]

    # the relevant neurons are the ones to be pulled next
    # cb[relevant_neurons] = confidence_bounds(samples[relevant_neurons], variances[relevant_neurons], delta)
    relevant_neurons = set([i for i in range(filt_len) if shapley_values[i] + cb[i] > threshold and shapley_values[i] - cb[i] < threshold])

    return len(relevant_neurons)

delta = 0.05
k = 10
while update_once(delta, k) > 1:
    pass

neurons = np.partition(shapley_values, -k)[-k:]
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

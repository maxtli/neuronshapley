# %%
import numpy as np
import math
from tqdm import tqdm
import pickle

from ablation import filt_len, load_conv_means, forward_pass, genders_loader

# %%

def load_shapley():
    with open("shapley_values.pkl", "rb") as pickle_file:
        shapley_values = pickle.load(pickle_file)
    with open("variances.pkl", "rb") as pickle_file:
        variances = pickle.load(pickle_file)
    with open("cb.pkl", "rb") as pickle_file:
        cb = pickle.load(pickle_file)
    with open("samples.pkl", "rb") as pickle_file:
        samples = pickle.load(pickle_file)
    with open("iterations.pkl", "rb") as pickle_file:
        i = pickle.load(pickle_file)
    return shapley_values, variances, cb, samples, i

shapley_values, variances, cb, samples, i = load_shapley()

def init_shapley(iters=3000, load=True):
    truncate_threshold = 0.52
    epsilon = 0.001
    delta = 0.1
    k = 100

    if load:
        shapley_values, variances, cb, samples, i = load_shapley()
        relevant_neurons = compute_relevant_neurons(shapley_values, cb, epsilon, k)
    else:
        shapley_values = np.zeros(filt_len)
        variances = np.zeros(filt_len)

        # # confidence bounds are symmetric: the intervals are [shapley_values[i] +- cb[i]]
        cb = np.zeros(filt_len)
        samples = np.zeros(filt_len)
        i = 0
        relevant_neurons = set(range(filt_len))

    max_loss = 1
    conv_means = load_conv_means()
    all_neuron_acc = forward_pass(np.zeros(filt_len), conv_means, genders_loader)

    for j in range(iters):
        relevant_neurons = update_once(shapley_values, variances, cb, samples, truncate_threshold, epsilon, delta, k, relevant_neurons, max_loss, all_neuron_acc, conv_means)
        num_relevant = len(relevant_neurons)
        if j % 5 == 0:
            with open("iterations.pkl", "wb") as pickle_file:
                pickle.dump(i + j, pickle_file)
            with open("shapley_values.pkl", "wb") as pickle_file:
                pickle.dump(shapley_values, pickle_file)
            with open("variances.pkl", "wb") as pickle_file:
                pickle.dump(variances, pickle_file)
            with open("cb.pkl", "wb") as pickle_file:
                pickle.dump(cb, pickle_file)
            with open("samples.pkl", "wb") as pickle_file:
                pickle.dump(samples, pickle_file)

        if num_relevant == 1:
            break

    neurons = np.sort(np.partition(shapley_values, -k)[-k:])
    print(neurons)

def compute_relevant_neurons(shapley_values, cb, epsilon, k):
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
    return relevant_neurons

# shapley_values, variances, cb, samples
def update_once(shapley_values, variances, cb, samples, truncate_threshold, epsilon, delta, k, relevant_neurons, max_loss, all_neuron_acc, conv_means):
    seq = np.random.permutation(np.arange(filt_len))

    # form ablation array, where 0 -> don't ablate, and 1 -> ablate.
    ablate_mask = np.zeros(filt_len)

    prev_loss = all_neuron_acc

    def update_loss():
        # start = time.time()
        out = (
            forward_pass(ablate_mask, conv_means, genders_loader)
            if prev_loss >= truncate_threshold
            else prev_loss
        )
        # print("update loss time", time.time() - start)
        return out

    # remove neurons by adding them to the ablate mask
    for k, neuron in enumerate(tqdm(seq)):
        ablate_mask[neuron] = 1
        if neuron not in relevant_neurons:
            if k < len(seq) - 1 and seq[k + 1] in relevant_neurons:
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

        if k % 30 == 0:
            print("acc", prev_loss)

        # I think ghorbani forgets to do a union bound here, twice.
        # First, the pulls are not independent (previous pulls influence which future pulls to do).
        # Second, you want to guarantee that wp 1-delta, ALL confidence bounds are valid, not just 1.
        # original code:
        #     cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / delta) / counts[counts > 1]) +\
        # 7/3 * R * np.log(2 / delta) / (counts[counts > 1] - 1)
        cb[neuron] = (
            math.sqrt(2 * variances[neuron] * math.log(2 / delta) / samples[neuron])
            + 7 / 3 * max_loss * math.log(2 / delta) / (samples[neuron] - 1)
            if samples[neuron] > 1
            else max_loss
        )

        prev_loss = loss

    return compute_relevant_neurons(shapley_values, cb, epsilon, k)
# %%

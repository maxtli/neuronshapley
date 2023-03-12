# %%
from ablation import filt_len, celeba_val_loader, load_conv_means, forward_pass, get_gender_dataloaders

import numpy as np
from tqdm import tqdm
import pickle


# %%
# load in shapley values
with open("shapley_values.pkl", "rb") as pickle_file:
    shapley_values = pickle.load(pickle_file)

shapley_values.sort()
print(sum(shapley_values))
print(sum(shapley_values[-100:]))

# %%

# batch is entire dataset
white_men_dataloader, white_women_dataloader, black_men_dataloader, black_women_dataloader = get_gender_dataloaders()

with open("shapley_values_first_iter.pkl", "rb") as pickle_file:
    shapley_values = pickle.load(pickle_file)

neurons_sorted = np.argsort(shapley_values)  # ascending order


def check_score(ablate_mask):
    conv_means = load_conv_means()
    wm_acc = forward_pass(ablate_mask, conv_means, white_men_dataloader)
    ww_acc = forward_pass(ablate_mask, conv_means, white_women_dataloader)
    bm_acc = forward_pass(ablate_mask, conv_means, black_men_dataloader)
    bw_acc = forward_pass(ablate_mask, conv_means, black_women_dataloader)
    overall_acc = sum([wm_acc, ww_acc, bm_acc, bw_acc]) / 4
    # print("White men", wm_acc)
    # print("White women", ww_acc)
    # print("Black men", bm_acc)
    # print("Black women", bw_acc)
    # print("overall", overall_acc)
    # print(len(celeba_val_loader))
    celeba_acc = forward_pass(
        ablate_mask, conv_means, celeba_val_loader
    )  # TODO uses new batch every iteration, weird
    # print("celeba", celeba_acc)

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

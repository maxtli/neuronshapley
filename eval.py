# %%
# Evaluation file for ablating neurons selected by highest Shapley values

from ablation import filt_len, celeba_val_loader, load_conv_means, forward_pass, get_gender_dataloaders

import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# %%
# # load in shapley values
# with open("shapley_values.pkl", "rb") as pickle_file:
#     shapley_values = pickle.load(pickle_file)

# shapley_values.sort()
# print(sum(shapley_values))
# print(sum(shapley_values[-100:]))

# batch is entire dataset
white_men_dataloader, white_women_dataloader, black_men_dataloader, black_women_dataloader = get_gender_dataloaders()

# %%

# check the score on FairFace for a given ablation of the filters
def check_score(ablate_mask, conv_means):
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
    celeba_acc = 0
    # celeba_acc = forward_pass(
    #     ablate_mask, conv_means, celeba_val_loader
    # )
    # print("celeba", celeba_acc)

    return wm_acc, ww_acc, bm_acc, bw_acc, overall_acc, celeba_acc

# plot accuracy vs number of top filters ablated
def save_plots(iters=False):
    conv_means = load_conv_means()

    with open(f"shap_records/shapley_values/{iters}.pkl" if iters else "shapley_values.pkl", "rb") as pickle_file:
        shapley_values = pickle.load(pickle_file)

    neurons_sorted = np.argsort(shapley_values)  # ascending order

    # print(sum(shapley_values[neurons_sorted[-100:]]))
    # print(shapley_values[neurons_sorted[-1]])

    ablate_mask = np.zeros(filt_len)
    
    check_score(ablate_mask, conv_means)
    wm_accs, ww_accs, bm_accs, bw_accs, overall_accs, celeba_accs = [], [], [], [], [], []
    for i in tqdm(range(30)):
        # print("shapley value", shapley_values[neurons_sorted[-i - 1]])
        ablate_mask[neurons_sorted[-i - 1]] = 1
        wm_acc, ww_acc, bm_ac, bw_acc, overall_acc, celeba_acc = check_score(ablate_mask, conv_means)
        wm_accs.append(wm_acc)
        ww_accs.append(ww_acc)
        bm_accs.append(bm_ac)
        bw_accs.append(bw_acc)
        overall_accs.append(overall_acc)
        celeba_accs.append(celeba_acc)
    
    plt.hist(shapley_values, bins=100)
    plt.show()
    plt.savefig(f"pics/dist/{iters}.png")

    # plt.plot(celeba_accs, label="celeba")
    plt.plot(wm_accs, label="white men")
    plt.plot(ww_accs, label="white women")
    plt.plot(bm_accs, label="black men")
    plt.plot(bw_accs, label="black women")
    plt.plot(overall_accs, label="overall")
    plt.legend()
    # plt.ylim(.5, 1.05)
    plt.xlabel("Number of filters ablated")
    plt.ylabel("Test Accuracy (%)")
    plt.show()
    plt.savefig(f"pics/accs/{iters}.png")

# %%
if __name__ == "__main__":
    save_plots()

# %%

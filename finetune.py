# %%

from model import get_squeezenet, epoch
from datasets import get_celeb_data_loaders, get_combined_gender_loader, get_gender_dataloaders
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import matplotlib.pyplot as plt
import torch as t

# %%

device = "cuda"
model = get_squeezenet("squeezenet.pth", device=device)

# %% 
celeb_train, celeb_val = get_celeb_data_loaders(batch_size=64)
gender_loader = get_combined_gender_loader(batch_size=64, num_images_each=200, val=False)
gender_loader_val = get_combined_gender_loader(batch_size=64, num_images_each=200, val=True)
white_men, white_women, black_men, black_women = get_gender_dataloaders(batch_size=64, num_images_each=200, val=True)

# %%

criterion = CrossEntropyLoss()

epoch(model, celeb_val, criterion, None, device, name="CelebA Val")
train_accs = [epoch(model, gender_loader, criterion, None, device, name="Combined Gender Train")] 
val_accs = [epoch(model, gender_loader_val, criterion, None, device, name="Combined Gender Val")]
black_women_accs = [epoch(model, black_women, criterion, None, device, name="Black Women Acc (val)")]

# %%

epochs = 50
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for ep in range(epochs):
    train_accs.append(epoch(model, gender_loader, criterion, optimizer, device, name=f"combined gender, epoch {ep}"))
    val_accs.append(epoch(model, gender_loader_val, criterion, None, device, name=f"combined gender val, epoch {ep}"))
    black_women_accs.append(epoch(model, black_women, criterion, None, device, name=f"black women val, epoch {ep}"))

# %% 

plt.plot(train_accs, label="train")
plt.plot(val_accs, label="val")
plt.plot(black_women_accs, label="black women (val)")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Finetuning (200 images per gender/race)")
plt.show()

# %%

epoch(model, celeb_val, CrossEntropyLoss(), None, device, name="CelebA Val")
epoch(model, gender_loader, CrossEntropyLoss(), None, device, name="Combined Gender")
# %%

white_men, white_women, black_men, black_women = get_gender_dataloaders(val=True)

epoch(model, white_men, CrossEntropyLoss(), None, device, name="White Men")
epoch(model, white_women, CrossEntropyLoss(), None, device, name="White Women")
epoch(model, black_men, CrossEntropyLoss(), None, device, name="Black Men")
epoch(model, black_women, CrossEntropyLoss(), None, device, name="Black Women")

white_men, white_women, black_men, black_women = get_gender_dataloaders(val=False)

epoch(model, white_men, CrossEntropyLoss(), None, device, name="Train White Men")
epoch(model, white_women, CrossEntropyLoss(), None, device, name="Train White Women")
epoch(model, black_men, CrossEntropyLoss(), None, device, name="Train Black Men")
epoch(model, black_women, CrossEntropyLoss(), None, device, name="Train Black Women")
# %%

# get gradient norm of each filter on gender_loader

t.grad_enabled = True
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

# get all conv layers (where we'll be ablating)
convs = list(filter(lambda x: isinstance(x, t.nn.Conv2d), list(model.modules())))

# get total number of conv filters
filt_len = sum([c.out_channels for c in convs])

# setup dictionary to store gradients
grads = dict()

for data, labels in gender_loader:
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()

    for conv in convs:
        for i in range(conv.out_channels):
            grads[(conv, i)] = grads.get((conv, i), 0) + conv.weight.grad[i].detach()

# dictionary for gradient norms of average of grads
grad_weight_prod = dict()
for k in grads:
    # grad_weight_prod[k] = grads[k].norm().item()
    grad_weight_prod[k] = grads[k] / len(gender_loader)
    dot_with_weight = t.einsum("ijk, ijk -> ijk", grad_weight_prod[k], k[0].weight[k[1]])
    dot_with_weight = t.einsum("ijk -> ", dot_with_weight).item()
    grad_weight_prod[k] = dot_with_weight


for data, labels in celeb_val:
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()

    for conv in convs:
        for i in range(conv.out_channels):
            grads[(conv, i)] = grads.get((conv, i), 0) + conv.weight.grad[i].detach()

# dictionary for gradient norms of average of grads
grad_weight_prod_celeb = dict()
for k in grads:
    # grad_weight_prod[k] = grads[k].norm().item()
    grad_weight_prod_celeb[k] = grads[k] / len(celeb_val)
    dot_with_weight = t.einsum("ijk, ijk -> ijk", grad_weight_prod_celeb[k], k[0].weight[k[1]])
    dot_with_weight = t.einsum("ijk -> ", dot_with_weight).item()
    grad_weight_prod_celeb[k] = dot_with_weight

# %% 

# get top 10 filters to ablate
from tqdm import tqdm
from ablation import load_conv_means, forward_pass
import numpy as np

ordered_list = [] 
idxs = []
for i, conv in enumerate(tqdm(convs)):
    for j in range(conv.out_channels):
        ordered_list.append(grad_weight_prod_celeb[(conv, j)] - grad_weight_prod[(conv, j)])
        idxs.append((conv, j))

top_idx = np.argsort(ordered_list)[:10]

# %%

conv_means = load_conv_means()
ablate_mask = np.zeros(filt_len)

# train_accs = [epoch(model, gender_loader, criterion, None, device, name="Combined Gender Train")] 
# val_accs = [epoch(model, gender_loader_val, criterion, None, device, name="Combined Gender Val")]
white_women_accs = [epoch(model, white_women, criterion, None, device, name="White Women Acc (val)")]
white_men_accs = [epoch(model, white_men, criterion, None, device, name="White Men Acc (val)")]
black_women_accs = [epoch(model, black_women, criterion, None, device, name="Black Women Acc (val)")]
black_men_accs = [epoch(model, black_men, criterion, None, device, name="Black Men Acc (val)")]



# ablate filters with smallest grad_weight_prod
for i in tqdm(top_idx):
    ablate_mask[i] = 1
    # train_accs.append(forward_pass(ablate_mask, conv_means, gender_loader, zero_ablate=False))
    # val_accs.append(forward_pass(ablate_mask, conv_means, gender_loader_val, zero_ablate=False))
    black_women_accs.append(forward_pass(ablate_mask, conv_means, black_women, zero_ablate=False))
    black_men_accs.append(forward_pass(ablate_mask, conv_means, black_men, zero_ablate=False))
    white_women_accs.append(forward_pass(ablate_mask, conv_means, white_women, zero_ablate=False))
    white_men_accs.append(forward_pass(ablate_mask, conv_means, white_men, zero_ablate=False))

    

plt.plot(train_accs, label="train")
plt.plot(val_accs, label="val")
plt.plot(black_women_accs, label="black women (val)")
plt.legend()
plt.xlabel("Filters Removed")
plt.ylabel("Accuracy")
plt.title("Filter Removing (200 images per gender/race, zero ablation)")
plt.show()
# %%

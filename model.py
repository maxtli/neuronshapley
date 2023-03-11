from torchvision.models import squeezenet1_1
import torch
from tqdm import tqdm


def get_squeezenet(weights_path=None, device="cpu"):
    model = squeezenet1_1(pretrained=True)
    model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 2
    if weights_path:
        print("Loading weights from {}".format(weights_path))
        model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    return model


# training/val loop
def epoch(model, loader, criterion, optimizer, device, verbose=True):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    # turn off gradients if not in train
    with torch.set_grad_enabled(train):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(tqdm(loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            # if labels is an int, convert to tensor
            if isinstance(labels, int):
                labels = torch.tensor(labels)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels.data) / len(labels)
            if i % 1000 == 999 or i == len(loader) - 1:
                tqdm.write(
                    "{} Iteration: {}, Loss: {:.4f}, Acc: {:.4f}".format(
                        "TRAIN" if train else "VAL",
                        i + 1,
                        running_loss / (i + 1 if i == len(loader) - 1 else i + 1),
                        running_acc / (i + 1 if i == len(loader) - 1 else i + 1),
                    )
                )
                running_loss = 0.0
                running_acc = 0.0

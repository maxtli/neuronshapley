import pandas as pd
import torchvision.transforms as transforms
import PIL
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class CelebDataset(Dataset):
    """
    Load the CelebA dataset for gender detection task.
    """

    def __init__(self, img_dir, transform, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = list(sorted(os.listdir(img_dir)))
        self.transform = transform
        df = pd.read_csv("data/celeba/list_attr_celeba.csv", index_col="image_id")
        self.genders = df["Male"]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = PIL.Image.open(img_path)
        label = int(
            (self.genders[self.img_labels[idx]] + 1) / 2
        )  # convert from -1,1 to 0, 1 labels
        image = self.transform(image)
        return image, label


class FairfaceDataset(Dataset):
    """
    Load the Fairface dataset for balanced gender detection.
    """

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


transform = transforms.Compose(
    [
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def get_celeb_data_loaders(batch_size=32, num_workers=2):
    train_set = CelebDataset(img_dir="data/celeba/train", transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_set = CelebDataset(img_dir="data/celeba/val", transform=transform)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def get_gender_datasets(num_images=200, val=False):
    # Load the fairface dataset, filtered to be in 20-59 age range and only white/black
    df = pd.read_csv("data/fairface/fairface_label_val.csv", index_col="file")
    ages = (
        (df["age"] != "3-9")
        * (df["age"] != "10-19")
        * (df["age"] != "60-69")
        * (df["age"] != "more than 70")
    )

    white_men = df.loc[(df["race"] == "White") * (df["gender"] == "Male") * ages]
    white_women = df.loc[(df["race"] == "White") * (df["gender"] == "Female") * ages]
    black_men = df.loc[(df["race"] == "Black") * (df["gender"] == "Male") * ages]
    black_women = df.loc[(df["race"] == "Black") * (df["gender"] == "Female") * ages]

    if val:
        white_men = white_men[-num_images:]
        white_women = white_women[-num_images:]
        black_men = black_men[-num_images:]
        black_women = black_women[-num_images:]
    else:
        white_men = white_men[:num_images]
        white_women = white_women[:num_images]
        black_men = black_men[:num_images]
        black_women = black_women[:num_images]

    white_men_ds = FairfaceDataset("data/fairface", transform, white_men)
    white_women_ds = FairfaceDataset("data/fairface", transform, white_women)
    black_men_ds = FairfaceDataset("data/fairface", transform, black_men)
    black_women_ds = FairfaceDataset("data/fairface", transform, black_women)

    return white_men_ds, white_women_ds, black_men_ds, black_women_ds


def get_gender_dataloaders(batch_size=128, num_images_each=200, num_workers=2, val=False):
    white_men_ds, white_women_ds, black_men_ds, black_women_ds = get_gender_datasets(
            num_images=num_images_each,
            val=val
        )

    white_men_loader = DataLoader(
        white_men_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    white_women_loader = DataLoader(
        white_women_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    black_men_loader = DataLoader(
        black_men_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    black_women_loader = DataLoader(
        black_women_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return white_men_loader, white_women_loader, black_men_loader, black_women_loader


def get_combined_gender_loader(batch_size=128, num_images_each=200, num_workers=2, val=False):
    white_men_ds, white_women_ds, black_men_ds, black_women_ds = get_gender_datasets(
        num_images=num_images_each,
        val=val
    )

    combined_dataset = ConcatDataset(
        [white_men_ds, white_women_ds, black_men_ds, black_women_ds]
    )
    return DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

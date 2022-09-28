import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


class THGStrainStressDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, label_transform=None):

        # header = 0, assume there is a header in the labels.csv file.
        self.img_labels = pd.read_csv(labels_file, header=0)
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()

        # Extract the labels from the indexed row, not including the index.
        labels = torch.tensor(self.img_labels.iloc[idx, 1:].values.astype(float))
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            labels = self.label_transform(labels)
        return image, labels


class RandomImageDataset(Dataset):
    def __init__(self, size):
        self.images = self.generate_random_image_set(size)
        self.img_labels = self.generate_random_label_set(size)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels.iloc[idx, [1, 2, 3]]
        return image, label

    @staticmethod
    def generate_random_image_set(size):
        """Generate random images. (Inspired by https://stackoverflow.com/a/10901092.)

        Returns:
            random_image_set (list[Image]): the randomly generated image set.
        """
        random_image_set = []
        for _ in range(size):
            random_array = np.random.rand(258, 258) * 255
            random_image = Image.fromarray(random_array.astype("uint8"))
            random_image_set.append(random_image)

        return random_image_set

    @staticmethod
    def generate_random_label_set(size):
        """Generate random labels.

        Returns:
            random_label_set (list[str])
        """
        random_label_set = []
        for id in range(size):
            label_1 = np.random.rand()
            label_2 = np.random.rand()
            label_3 = np.random.rand()
            random_label_set.append(f"{id},{label_1},{label_2},{label_3}")

        return random_label_set

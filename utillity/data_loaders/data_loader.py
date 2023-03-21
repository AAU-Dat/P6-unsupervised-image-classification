import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset





class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, torchvision.io.ImageReadMode.GRAY)
        label = self.img_labels.iloc[idx, 1]
        knn = self.img_labels.iloc[idx, 2]
        image = image.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



def dataloader():
    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transformer)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transformer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
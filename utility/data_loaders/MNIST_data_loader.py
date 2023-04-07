from torch.utils.data import Dataset
import pandas as pd
import torchvision
import torch
import os

root_dir = '../data'


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[0, 0])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.GRAY)
        image = image.to(torch.float32)
        print(image.shape)
        self.image_shape = image.shape


    def __len__(self):
        return len(self.img_labels)
    

    def shape(self):
        return self.image_shape


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.GRAY)
        label = self.img_labels.iloc[idx, 1]
        image = image.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def MNIST(batch_size):
    transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root=root_dir, train=True, download=True, transform=transformer)
    test_dataset = torchvision.datasets.MNIST(root=root_dir, train=False, download=True, transform=transformer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, [1,28,28]


def custom_MNIST(batch_size, foldername):
    transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dir = root_dir + '/' + foldername

    train_dataset = CustomImageDataset(train_dir + '/data.csv', train_dir)
    test_dataset = torchvision.datasets.MNIST(root=root_dir, train=False, download=True, transform=transformer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

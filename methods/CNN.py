import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())

# Hyper-parameters
num_epochs = 4
batch_size = 1
learning_rate = 0.001
train_data = 'MNIST_allTransforms'


# Data setup
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
        return image, label, knn


root_dir = './data/' + train_data
transformer = transforms.Compose([transforms.ToTensor()])

train_dataset = CustomImageDataset(root_dir + '/data.csv', root_dir)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformer)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ConvNet(nn.Module):
    def __init__(self, convolutions, pooling_size, layers_and_nodes):
        super(ConvNet, self).__init__()
        # import configs to tell which dataset we are running on
        mnist = [1, 28]
        temp = mnist
        convs = []
        for i in convolutions:
            convs = nn.Conv2d(temp[0], convolutions[i][0], convolutions[i][1])
            temp[1] = (temp[1] - ((convolutions[1]-1)/2))/pooling_size

        layers_nodes = []
        for i in layers_and_nodes:


        self.convs = convs
        self.pooling = nn.MaxPool2d(pooling_size, pooling_size)



        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 1, 28, 28
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 4, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 200
        x = F.relu(self.fc2(x))  # -> n, 100
        x = self.fc3(x)  # -> n, 10
        return x


model = ConvNet().to(device)


n_total_steps = len(train_loader)
for i, (images, labels, knn) in enumerate(train_loader):
    # origin shape: [batch_size, 1, 28, 28] = batch_size, 1, 784
    # input_layer: 1 input channels, 6 output channels, 5 kernel size
    images = images.to(device)
    outputs = model(images)
    print(outputs.shape)
    print(outputs)
    imshow(torchvision.utils.make_grid(images))
    print('hi')
    if (i + 1) % 2000 == 0:
        print(f'Step [{i + 1}/{n_total_steps}]')

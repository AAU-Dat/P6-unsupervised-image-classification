import numpy
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
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())

# Parameters
batch_size_train = 300000
batch_size_test = 10000
train_data = 'MNIST_allTransforms'
clusters = 60000


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
        return image, label


root_dir = '../data/' + train_data
transformer = transforms.Compose([transforms.ToTensor()])

# train_dataset = CustomImageDataset(root_dir + '/data.csv', root_dir)
train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transformer)
test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transformer)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ConvNet(nn.Module):
    def __init__(self, convolutions, pooling_size, layers):
        super(ConvNet, self).__init__()
        # temp = image shape 1 color channel 28x28 pixels
        temp = [1, 28]

        # convolutions is set acording to the temp and convolutions
        convs = []
        convs.append(nn.Conv2d(temp[0], convolutions[0][0], convolutions[0][1]))
        temp[1] = int((temp[1] - (convolutions[0][1] - 1)) / pooling_size)
        for i in range(1, len(convolutions)):
            temp[0] = convolutions[i][0]
            temp[1] = int((temp[1] - (convolutions[i][1] - 1)) / pooling_size)
            convs.append(nn.Conv2d(convolutions[i-1][0], convolutions[i][0], convolutions[i][1]))

        # layers is set acording to convolution outputs and layers
        layers_temp = []
        layers_temp.append(nn.Linear(temp[0] * int(temp[1]) * int(temp[1]), layers[0]))
        for i in range(1, len(layers)):
            layers_temp.append(nn.Linear(layers[i-1], layers[i]))

        self.output_from_convs = temp[0] * int(temp[1]) * int(temp[1])
        self.convs = convs
        self.pool = nn.MaxPool2d(pooling_size, pooling_size)
        self.layers = layers_temp


    def forward(self, x):
        for i in self.convs:
            x = self.pool(F.relu(i(x)))
        x = x.view(-1, self.output_from_convs)
        for i in self.layers:
            x = F.relu(i(x))
        return x


# convolution settings = [[output channeels, kernel size(must be uneven)]] where legth of the outer array is the number of convolutions
convolutions = [[3, 5], [16, 3]]
# pooling size is the size of the kernel
pooling_size = 2
# layers = [output variables]  where length of the outer array is the number of nn layers
layers = [120, 84]
model = ConvNet(convolutions, pooling_size, layers).to(device)
k_means = KMeans(n_clusters=clusters, n_init=5)


n_total_steps = len(train_loader)
for i, (images, labels) in enumerate(train_loader):
    # origin shape: [batch_size, 1, 28, 28] = batch_size, 1, 784
    # input_layer: 1 input channels, 6 output channels, 5 kernel size
    images = images.to(device)
    outputs = model(images)
    # k_nn = neighbors.NearestNeighbors(n_neighbors=2)
    # k_nn.fit(outputs.detach().numpy())
    # res = k_nn.kneighbors(outputs.detach().numpy(), 2, return_distance=True)
    k_means.fit(outputs.detach().numpy())

    if (i + 1) % 2000 == 0:
        print(f'Step [{i + 1}/{n_total_steps}]')
    break


# eval
eval = numpy.zeros((clusters, 10))
for i, (images, labels) in enumerate(test_loader):
    outputs = model(images)
    res = k_means.predict(outputs.detach().numpy())
    label = labels.numpy()
    for i in range(len(res)):
        eval[res[i]][label[i]] += 1
    break

#self label
reps = []
for i in range(clusters):
    reps.append(0)

for i in range(clusters):
    for j in range(10):
        if eval[i][j] > eval[i][reps[i]]:
            reps[i] = j


# find correct and wrong answers
right = []
for i in range(clusters):
    right.append(0)

wrong = []
for i in range(clusters):
    wrong.append(0)

for i in range(clusters):
    for j in range(10):
        if reps[i] != j:
            wrong[i] += eval[i][j]
        else:
            right[i] += eval[i][j]


# find accuracy
accuarcy = []
for i in range(clusters):
    accuarcy.append(0)

wrongs = 0
rights = 0
for i in range(10):
    wrongs += wrong[i]
    rights += right[i]
    if right[i] != 0 and (right[i] + wrong[i]) != 0:
        accuarcy[i] = right[i]/(right[i] + wrong[i])

print(eval)
print(reps)
print(accuarcy)
print(rights)
print(wrongs)
if rights != 0 and (rights + wrongs) != 0:
    print(rights/(rights + wrongs))

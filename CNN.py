import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#data setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

train_dataset = torchvision.datasets.MNIST(root='\data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(root='\data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[j]:5s}' for j in labels.numpy()))


# data augmentation
# add images to seperate dataset
# mearge datasets

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        # -> n, 1, 28, 28
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 4, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 4, 4
        x = x.view(-1, 16 * 4 * 4)  # -> n, 256
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        return x


model = ConvNet().to(device)

n_total_steps = len(train_loader)
for i, (images, labels) in enumerate(train_loader):
    # origin shape: [batch_size, 1, 28, 28] = batch_size, 1, 784
    # input_layer: 3 input channels, 6 output channels, 5 kernel size
    images = images.to(device)

    outputs = model(images)

    if (i + 1) % 2000 == 0:
        print(f'Step [{i + 1}/{n_total_steps}]')


print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)






import matplotlib.pyplot as plt
import torchvision
import torch
import os
from torchvision import transforms
import torchvision.transforms as T


batch_size = 1
# dashes for the op system
ops = '/'
image_name = 1
num_of_augments = 4

parentParth = '.' + ops + 'data'
folderParths = ['rotated', 'cropped', 'blurred', 'allTransforms']
transformations = [T.RandomRotation(degrees=(-85, 85)), T.RandomCrop((28, 28), padding=4), transforms.GaussianBlur(3)]

# load data
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if not os.path.exists(parentParth):
    os.makedirs(parentParth)

for i in range(len(folderParths)):
    folderParths[i] = parentParth + ops + folderParths[i]

for i in folderParths:
    if not os.path.exists(i):
        os.makedirs(i)


# generate and store images
n_total_steps = len(train_loader)
for i, (images, labels) in enumerate(train_loader):

    # saves the original
    for p in folderParths:
        plt.imsave(p + ops + str(image_name) + '.png', images[0][0], cmap='gray')
    image_name += 1

    # saves augments
    tempimg = images
    for j in range(num_of_augments):
        for t in range(len(transformations)):
            plt.imsave(folderParths[t] + ops + str(image_name) + '.png', transformations[t](images)[0][0], cmap='gray')
            tempimg = transformations[t](tempimg)

        plt.imsave(folderParths[len(folderParths) - 1] + ops + str(image_name) + '.png', tempimg[0][0], cmap='gray')
        image_name += 1
        plt.close()
    print('hi')
    # rotate image
    # crop image
    # blur image
    # do all three things to an image
    # save all images in their directories
    # add ome to image number counter

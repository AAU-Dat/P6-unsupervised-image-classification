from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch
import csv
import os


batch_size = 1
# dashes for the op system
ops = '/'
image_name = 1
num_of_augments = 4

parentParth = '.' + ops + 'data'
folderParths = ['rotated', 'cropped', 'blurred', 'allTransforms']
transformations = [transforms.RandomRotation(degrees=(-85, 85)), transforms.RandomCrop((28, 28), padding=8), transforms.GaussianBlur(5)]

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


csv_files = []
for i in range(len(folderParths)):
    csv_file = open(folderParths[i] + ops + 'data.csv', 'w')
    csv_file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    csv_files.append([csv_file, csv_file_writer])


# generate and store images
n_total_steps = len(train_loader)
for i, (images, labels) in enumerate(train_loader):

    # saves the original
    for p in range(len(folderParths)):
        name = str(image_name) + '.png'
        plt.imsave(name, images[0][0], cmap='gray')
        csv_files[p][1].writerow([name, '', ''])

    image_name += 1

    if image_name % 10000 == 0:
        print(str(image_name) + " / " + str(60000 * num_of_augments + 1))

    # saves augments
    for j in range(num_of_augments):
        tempimg = images
        for t in range(len(transformations)):
            name = str(image_name) + '.png'
            plt.imsave(folderParths[t] + ops + str(image_name) + '.png', transformations[t](images)[0][0], cmap='gray')
            csv_files[t][1].writerow([name, '', ''])
            tempimg = transformations[t](tempimg)

        name = str(image_name) + '.png'
        plt.imsave(name, tempimg[0][0], cmap='gray')
        csv_files[len(folderParths) - 1][1].writerow([name, '', ''])
        image_name += 1
        plt.close()

for i in range(len(folderParths)):
    csv_files[i][0].close()

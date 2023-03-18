from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch
import csv
import os

# how many images we load from memory at a time
batch_size = 60000
# the conter that decides what the image is called
image_name = 1
# number of random augments per image
num_of_augments = 4
# name of the dataset (should match with the dataset pulled in in train datasets)
dataset = '/MNIST'
# the image type in the database, according to imsave()
image_type = 'gray'

# where all data should be stored
parentParth = '../data'
# folder endpoint for each transformation (index should match with transformation array)
folderParths = ['_rotated', '_cropped', '_blurred', '_allTransforms']
# transformations (index should match with folderparths array)
transformations = [transforms.RandomRotation(degrees=(-85, 85)), transforms.RandomCrop((28, 28), padding=8), transforms.GaussianBlur(5)]

# the basic transformation for the image when loaded from memory
transform = transforms.Compose([transforms.ToTensor()])

# the dataset downloaded (should match with the dataset name in the variable dataset)
train_dataset = torchvision.datasets.MNIST(root=parentParth, train=True, download=True, transform=transform)






train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# generates the full parth to each dataset
for i in range(len(folderParths)):
    folderParths[i] = parentParth + dataset + folderParths[i]

# checks if the data folder exists and makes it
if not os.path.exists(parentParth):
    os.makedirs(parentParth)

# checks if the dataset folders exists and makes it
for i in folderParths:
    if not os.path.exists(i):
        os.makedirs(i)


# opens and makes a writer to all csv files
csv_files = []
for i in range(len(folderParths)):
    csv_file = open(folderParths[i] + '/data.csv', 'w')
    csv_file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    csv_files.append([csv_file, csv_file_writer])


# generate and store images
for i, (images, labels) in enumerate(train_loader):

    # loops through the batches
    for batch_number in range(len(images)):

        # save original pictures
        for p in range(len(folderParths)):
            name = str(image_name) + '.png'
            plt.imsave(folderParths[p] + name, images[batch_number].squeeze().numpy(), cmap=image_type)
            csv_files[p][1].writerow([name, '', ''])

        image_name += 1

        # save augmented pictures
        for j in range(num_of_augments):
            tempimg = images[batch_number]

            #saves all single type augments
            for t in range(len(transformations)):
                name = str(image_name) + '.png'
                plt.imsave(folderParths[t] + name, transformations[t](images[batch_number]).squeeze().numpy(), cmap=image_type)
                csv_files[t][1].writerow([name, '', ''])
                tempimg = transformations[t](tempimg)

            # saves pictures with all types of augments
            name = str(image_name) + '.png'
            plt.imsave(folderParths[len(folderParths) - 1] + name, tempimg.squeeze().numpy(), cmap=image_type)
            csv_files[len(folderParths) - 1][1].writerow([name, '', ''])
            image_name += 1
            plt.close()

# closes files
for i in range(len(folderParths)):
    csv_files[i][0].close()

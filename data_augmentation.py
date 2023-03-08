import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import csv


# Make sure that GPU works if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Download mnist dataset
mnist_dataset_downloaded = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print("Length of mnist dataset: " + str(len(mnist_dataset_downloaded)) + " images.")

# Create a loader for the mnist dataset without the labels
mnist_dataset_loader = torch.utils.data.DataLoader(mnist_dataset_downloaded, batch_size=1, shuffle=False)
print("Length of mnist dataset loader: " + str(len(mnist_dataset_loader)) + " images.")


# Create a function that have data augmentation features
def data_augmentation(image, label, i):
    # Add rotate the image
    if i == 'data_augmentation_rotation' or i == 'data_augmentation_all_together':
        image = transforms.RandomRotation(360)(image)

    # Add blur to the image
    if i == 'data_augmentation_blur' or i == 'data_augmentation_all_together':
        image = transforms.GaussianBlur(3)(image)

    # Add crop to the image
    if i == 'data_augmentation_crop' or i == 'data_augmentation_all_together':
        image = transforms.RandomCrop((28, 28), padding=4)(image)

    # Return the image and label
    return image, label


folder_names = ['data_augmentation_rotation', 'data_augmentation_blur', 'data_augmentation_crop',
                'data_augmentation_all_together']

# for loop that creates a folder and use the code from 48 to 77 in each of the loops
for i in folder_names:
    if not os.path.exists('./' + i):
        os.makedirs('./' + i)

    # Print the folder name
    print("\nFolder: " + i)

    # Loop through the mnist dataset loader
    for k, (image, label) in enumerate(mnist_dataset_loader):

        # Print the progress in procent
        print("\rProgress: " + str(round(k / len(mnist_dataset_loader) * 100, 2)) + "%", end='')

        # Save the original image to the i folder
        plt.imsave(i + '/' + str(k) + '_original.png', image[0][0], cmap='gray')

        # Create cvs file and add the original image to it and two comma after the image s
        with open(i + '.csv', 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([str(k) + '_original.png', '', ''])

        # Create x amount of new image on the original image and save them without labels
        for j in range(3):
            # Create a new image with data augmentation features
            new_image, new_label = data_augmentation(image, label, i)

            # Save the augmented image to the i folder
            plt.imsave(i + '/' + str(k) + '_' + str(j) + '.png', new_image[0][0], cmap='gray')

            # Add the new image to the csv file
            with open(i + '.csv', 'a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow([str(k) + '_' + str(j) + '.png', '', ''])

        # Break the loop if the amount of images is x
        #if k == 100:
        #    break
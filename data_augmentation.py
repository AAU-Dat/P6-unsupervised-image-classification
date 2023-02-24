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

# Create a loader for the mnist dataset
mnist_dataset_loader = torch.utils.data.DataLoader(mnist_dataset_downloaded, batch_size=1, shuffle=True)
print("Length of mnist dataset loader: " + str(len(mnist_dataset_loader)) + " batches.")

# Create a function that have data augmentation features
def data_augmentation(image, label):
    # Rotate the image
    image = transforms.RandomRotation(360)(image)

    # Add blur to the image
    image = transforms.GaussianBlur(3)(image)

    # Add crop to the image
    image = transforms.RandomCrop(28, padding=4)(image)

    # Return the image and label
    return image, label

# Create a folder to save the data augmentation images
if not os.path.exists('./data_augmentation'):
    os.makedirs('./data_augmentation')

"""
# Create a csv file to save all image
#with open('./data_augmentation.csv', 'w') as f:
"""

"""
# Create a numpy array to save the images
image_array = np.zeros((1, 28, 28))
"""

# Loop through the mnist dataset loader
for i, (image, label) in enumerate(mnist_dataset_loader):
    # Use GPU if possible
    image = image.to(device)
    label = label.to(device)

    # Save the original image
    plt.imsave('./data_augmentation/' + str(i) + '_original.png', image[0][0], cmap='gray')

    # Create x amount of new image on the original image and save them
    for j in range(1):
        # Create a new image
        new_image, new_label = data_augmentation(image, label)

        # Save the new image
        plt.imsave('./data_augmentation/' + str(i) + '_' + str(j) + '.png', new_image[0][0], cmap='gray')

    # Print the progress
    print("Progress: " + str(i + 1) + "/" + str(len(mnist_dataset_loader)) + " images.")

    # Break the loop if the amount of images is 10
    if i == 4:
        break

"""
# Convert the numpy array to a csv file
np.savetxt('./data_augmentation.csv', image_array, delimiter=',')
"""
from utility.data_loaders.MNIST_data_loader import *
from torchvision import transforms
from models.CNN import *
from numpy import *
from utility.loss_functions import *
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())


batch_size = 60000

transforms = torch.nn.Sequential(
    transforms.RandomRotation(40),
    transforms.RandomCrop((28, 28), padding=2)
)

# phi_theta network settings
# [outputs, kernal_size]
phi_theta_convolutions = [[10, 5], [25, 3]]
# kernal_size
phi_theta_pooling_size = 2
# number of nodes
phi_theta_layers = [150, 100, 50]
image_shape = [1, 28]




def main():
    # dataset
    train_loader, test_loader = MNIST(batch_size)
    train_loader.shape()


    #find nerest neigbors 
    phi_theta = CNN(phi_theta_convolutions, phi_theta_pooling_size, phi_theta_layers, train_loader.shape()).to(device)

    optimizer = torch.optim.SGD(phi_theta.parameters(), lr=0.01, momentum=0.5)
    criterion = Euclidian()


    # train the model
    n_total_steps = len(train_loader)
    print(f"\r0 out of {n_total_steps}", end="")
    for i, (images, labels) in enumerate(train_loader):

        t_images = transforms(images)
        t_output = phi_theta(t_images)

        output = phi_theta(images)
        
        loss = criterion(output, t_output)
        loss.backward()
        optimizer.step()
        
        print(f"\r{i} out of {n_total_steps}", end="")
    print("\n phi_theta trained \n")



    train_loader.dataset.__getitem__(1)

    # Optimize SCAN

    # selflabel



if __name__ == "__main__":
    main()

    
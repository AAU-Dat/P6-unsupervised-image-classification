from utility.data_loaders.MNIST_data_loader import *
from torchvision import transforms
from models.CNN import *
from numpy import *
from utility.loss_functions import *
import torch
import torch.optim as optim

# CUDNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())

batch_size = 60000

transforms = torch.nn.Sequential(
    transforms.RandomRotation(40),
    transforms.RandomCrop((28, 28), padding=2)
)

def main():
    # Parameters/Varibels shit

    # Data
    train_loader, test_loader, image_shape = MNIST(batch_size)

    # Model
    phi_theta = get_phi_theta_network(train_loader)

    # Optimizer
    optimizer = optim.Adam(phi_theta.parameters(), lr=0.1)

    # Warning
    

    # Loss function
    # Checkpoint
    # Main loop
    # Adjust lr
    # Train
    # Evaluate
    # Evaluate and save the final model
    # dataset


    # get phi theta model
    get_phi_theta_network(train_loader)



if __name__ == "__main__":
    main()



def get_phi_theta_network(train_loader):
    # initialize model
    phi_theta = ConvNet.to(device)

    # optimizer + criterion
    optimizer = optim.Adam(phi_theta.parameters(), lr=0.1)
    criterion = Euclidian()

    # train the model
    n_total_steps = len(train_loader)
    print(f'\r0 out of {n_total_steps}', end='')
    for i, (images, labels) in enumerate(train_loader):
        # make transformed images and run the network on them
        t_images = transforms(images)
        t_output = phi_theta(t_images)

        # run the network on the original network
        output = phi_theta(images)

        # optimize the network for the criterion
        loss = criterion(output, t_output)
        loss.backward()
        optimizer.step()
        print(f'{i}/{n_total_steps}')

    return phi_theta

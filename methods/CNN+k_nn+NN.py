import numpy
import torchvision.transforms

from utillity.data_loaders.MNIST_data_loader import *
from sklearn.neighbors import NearestNeighbors
from utillity.run_model import *
from models.CNN import *
from models.NN import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())

# Parameters
'!!!WARNING!!!'
# this method MUST train on the whole dataset at once
batch_size = 50
'!!!WARNING!!!'
train_data = 'MNIST_allTransforms'
clusters = 10

# CNN settings
# [outputs, kernal_size]
convolutions = [[10, 5], [25, 3]]
# kernal_size
pooling_size = 2
# number of nodes
cnn_layers = [150, 100, 50]
image_shape = [1, 28]

# knn
knn = NearestNeighbors()

# NN
# number of nodes
nn_layers = [150, 100, 50]


class CNN_knn_NN():
    def __init__(self, convolutions, pooling_size, layers_cnn, image_shape, knn, layers_nn):
        self.cnn = CNN(convolutions, pooling_size, layers_cnn, image_shape).to(device)
        self.knn = knn
        self.nn = NN(layers_nn, image_shape)

    def fit(self, images):
        output = self.cnn(images)
        self.knn.fit(output.detach().numpy())
        knn = self.knn.kneighbors(output.detach.numpy())
        res = self.nn(output)


    def predict(self, images):
        output = self.cnn(images)
        return self.nn(output)


def main():
    # import data
    train_loader, test_loader = MNIST(batch_size)
    # train_loader, test_loader = custom_MNIST(batch_size, train_data)

    # initialize model
    model = CNN_knn_NN(convolutions, pooling_size, cnn_layers, image_shape, knn, nn_layers)

    # train model
    train_model(model, train_loader, device, break_after_2=True)

    # test and evaluate model
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()

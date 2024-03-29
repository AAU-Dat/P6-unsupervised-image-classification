from utillity.data_loaders.MNIST_data_loader import *
from sklearn.cluster import KMeans
from utillity.run_model import *
from models.CNN import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())

# Parameters
'!!!WARNING!!!'
# this method MUST train on the whole dataset at once
batch_size = 60000
'!!!WARNING!!!'
train_data = 'MNIST_allTransforms'
clusters = 10

# CNN settings
# [outputs, kernal_size]
convolutions = [[10, 5], [25, 3]]
# kernal_size
pooling_size = 2
# number of nodes
layers = [150, 100, 50]
image_shape = [1, 28]

# k_means
k_means = KMeans(n_clusters=clusters, n_init=10)


class CNN__k_means():
    def __init__(self, convolutions, pooling_size, layers, image_shape, k_means):
        self.cnn = CNN(convolutions, pooling_size, layers, image_shape).to(device)
        self.k_means = k_means

    def fit(self, images):
        output = self.cnn(images)
        self.k_means.fit(output.detach().numpy())

    def predict(self, images):
        output = self.cnn(images)
        return self.k_means.predict(output.detach().numpy())


def main():
    # import data
    train_loader, test_loader = MNIST(batch_size)
    # train_loader, test_loader = custom_MNIST(batch_size, train_data)

    # initialize model
    model = CNN__k_means(convolutions, pooling_size, layers, image_shape, k_means)

    # train model
    train_model(model, train_loader, device, break_after_2=True)

    # test and evaluate model
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()

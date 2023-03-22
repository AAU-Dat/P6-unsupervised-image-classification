from utillity.data_loaders.MNIST_data_loader import *
from sklearn.cluster import KMeans
from models.CNN import *
import numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())

# Parameters
batch_size = 200
train_data = 'MNIST_allTransforms'
clusters = 10

# import data
train_loader, test_loader = MNIST(batch_size)

# convolution settings = [[output channeels, kernel size(must be uneven)]] where legth of the outer array is the number of convolutions
convolutions = [[3, 5], [16, 3]]
# pooling size is the size of the kernel
pooling_size = 2
# layers = [output variables]  where length of the outer array is the number of nn layers
layers = [120, 84]
model = CNN(convolutions, pooling_size, layers).to(device)
k_means = KMeans(n_clusters=clusters, n_init=5)


n_total_steps = len(train_loader)
for i, (images, labels) in enumerate(train_loader):
    # origin shape: [batch_size, 1, 28, 28] = batch_size, 1, 784
    # input_layer: 1 input channels, 6 output channels, 5 kernel size
    images = images.to(device)
    outputs = model(images)
    # k_nn = neighbors.NearestNeighbors(n_neighbors=2)
    # k_nn.fit(outputs.detach().numpy())
    # res = k_nn.kneighbors(outputs.detach().numpy(), 2, return_distance=True)
    k_means.fit(outputs.detach().numpy())

    if (i + 1) % 2000 == 0:
        print(f'Step [{i + 1}/{n_total_steps}]')
    break


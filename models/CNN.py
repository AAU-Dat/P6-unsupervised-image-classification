import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, convolutions, pooling_size, layers, image_shape):
        super(CNN, self).__init__()
        # convolutions = [[output channeels, kernel size(must be uneven)]] where legth of the outer array is the number of convolutions
        # pooling size is the size of the kernel
        # layers = [output variables]  where length of the outer array is the number of nn layers
        # image_shape = [color channels, length and width] yes only quadradic images

        # convolutions is set acording to the image_shape and convolutions
        convs = []
        convs.append(nn.Conv2d(image_shape[0], convolutions[0][0], convolutions[0][1]))
        image_shape[1] = int((image_shape[1] - (convolutions[0][1] - 1)) / pooling_size)
        for i in range(1, len(convolutions)):
            image_shape[0] = convolutions[i][0]
            image_shape[1] = int((image_shape[1] - (convolutions[i][1] - 1)) / pooling_size)
            convs.append(nn.Conv2d(convolutions[i-1][0], convolutions[i][0], convolutions[i][1]))

        # layers is set acording to convolution outputs and layers
        layers_image_shape = []
        layers_image_shape.append(nn.Linear(image_shape[0] * int(image_shape[1]) * int(image_shape[1]), layers[0]))
        for i in range(1, len(layers)):
            layers_image_shape.append(nn.Linear(layers[i-1], layers[i]))

        self.output_from_convs = image_shape[0] * int(image_shape[1]) * int(image_shape[1])
        self.convs = convs
        self.pool = nn.MaxPool2d(pooling_size, pooling_size)
        self.layers = layers_image_shape

    def forward(self, x):
        for i in self.convs:
            x = self.pool(nn.functional.relu(i(x)))
        x = x.view(-1, self.output_from_convs)
        for i in self.layers:
            x = nn.functional.relu(i(x))
        return x


class ConvNet(nn.Module): # Make structure om til ReLU
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
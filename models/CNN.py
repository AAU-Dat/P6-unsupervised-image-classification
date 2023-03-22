import torch.nn as nn


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
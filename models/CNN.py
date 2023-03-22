import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, convolutions, pooling_size, layers):
        super(CNN, self).__init__()
        # temp = image shape 1 color channel 28x28 pixels
        temp = [1, 28]

        # convolutions is set acording to the temp and convolutions
        convs = []
        convs.append(nn.Conv2d(temp[0], convolutions[0][0], convolutions[0][1]))
        temp[1] = int((temp[1] - (convolutions[0][1] - 1)) / pooling_size)
        for i in range(1, len(convolutions)):
            temp[0] = convolutions[i][0]
            temp[1] = int((temp[1] - (convolutions[i][1] - 1)) / pooling_size)
            convs.append(nn.Conv2d(convolutions[i-1][0], convolutions[i][0], convolutions[i][1]))

        # layers is set acording to convolution outputs and layers
        layers_temp = []
        layers_temp.append(nn.Linear(temp[0] * int(temp[1]) * int(temp[1]), layers[0]))
        for i in range(1, len(layers)):
            layers_temp.append(nn.Linear(layers[i-1], layers[i]))

        self.output_from_convs = temp[0] * int(temp[1]) * int(temp[1])
        self.convs = convs
        self.pool = nn.MaxPool2d(pooling_size, pooling_size)
        self.layers = layers_temp

    def forward(self, x):
        for i in self.convs:
            x = self.pool(nn.functional.relu(i(x)))
        x = x.view(-1, self.output_from_convs)
        for i in self.layers:
            x = nn.functional.relu(i(x))
        return x
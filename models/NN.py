import torch.nn as nn


class NN(nn.Module):
    def __init__(self, layers, image_shape):
        super(NN, self).__init__()
        # layers = [output variables]  where length of the outer array is the number of nn layers
        # image_shape = [color channels, length and width] yes only quadradic images

        # layers is set acording to convolution outputs and layers
        layers_image_shape = []
        layers_image_shape.append(nn.Linear(image_shape[0] * int(image_shape[1]) * int(image_shape[1]), layers[0]))
        for i in range(1, len(layers)):
            layers_image_shape.append(nn.Linear(layers[i-1], layers[i]))

        self.layers = layers_image_shape

    def forward(self, x):
        for i in self.layers:
            x = nn.functional.relu(i(x))
        return x
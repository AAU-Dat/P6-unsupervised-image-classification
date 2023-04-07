import tensorflow as tf
import torch.nn as nn


class Euclidian(nn.Module):
    def init(self):
        super(CustomLoss, self).init()
    
    def forward(self, output, target):
        return tf.norm(output - target, axis=-1)
    


#not finished
class Scan_loss_function(nn.Module):
    def init(self):
        super(CustomLoss, self).init()
    
    def forward(self, output, target):
        return 1 #return loss here
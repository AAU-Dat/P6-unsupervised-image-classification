from utillity.data_loaders.MNIST_data_loader import *
from sklearn.cluster import KMeans
from utillity.run_model import *
from models.CNN import *
from numpy import *
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # runs on gpu
print(torch.cuda.is_available())


class CustomLoss(nn.Module):
    def init(self):
        super(CustomLoss, self).init()
    
    def forward(self, output, target):
        return self.euclidean_distance(output, target)
    
    def euclidean_distance(x, y):
        return tf.norm(x - y, axis=-1)
    



def main():
    # dataset
    train_loader, test_loader = MNIST(batch_size)
    
    #find nerest neigbors 

    # Optimize SCAN

    # selflabel


    
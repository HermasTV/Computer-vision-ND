## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #conv layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        #Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=4,stride=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        #Batch Normalization 
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.norm4 = nn.BatchNorm2d(num_features=256)
        self.norm5 = nn.BatchNorm1d(num_features=2048)
        #dropout
        self.drop=nn.Dropout(p=.5)
        
        ## Linear layers
        self.fc1 = nn.Linear(in_features=1024, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.elu(self.norm1(self.conv1(x))))
        x = self.pool1(F.elu(self.norm2(self.conv2(x))))
        x = self.pool2(F.elu(self.norm3(self.conv3(x))))
        x = self.pool2(F.elu(self.norm4(self.conv4(x))))
        
        
        
        #flatten for FC layers
        x = x.view(x.size(0), -1) 
        
        x = F.elu(self.drop(self.fc1(x)))
        x = self.drop(F.elu(self.norm5(self.fc2(x))))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

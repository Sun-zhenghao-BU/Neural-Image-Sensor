import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define C1 layer (1 input channel, 8 output channel, kernel size is 5)
        self.conv1 = nn.Conv2d(1, 12, 7, padding=3, padding_mode='replicate')
        # # Define a batchNorm layer
        # self.bn1 = nn.BatchNorm2d(12)
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define full connection layers size
        self.fc1 = nn.Linear(12 * 14 * 14, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network
        # Apply convolutional layer 1
        x = self.conv1(x)
        # Apply padding to the 8-channel output
        # # Apply batch normalization
        # x = self.bn1(x)
        # Apply activation function
        x = self.relu(x)
        # Apply max pooling
        x = self.maxPool1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.log_softmax(x, dim=1)

        return x

model = LeNet()
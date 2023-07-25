import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define C1 layer (1 input channel, 8 output channel, kernel size is 5)
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2, padding_mode='replicate')
        # Define a batchNorm layer
        self.bn1 = nn.BatchNorm2d(8)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define full connection layers size
        self.fc1 = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network
        x = self.maxPool1(self.relu(self.bn1(self.conv1(x))))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.log_softmax(x, dim=1)

        return x

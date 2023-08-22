import torch
import torch.nn as nn
import torch.nn.functional as func

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define full connection layers size
        self.fc1 = nn.Linear(1 * 100 * 100, 120)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network

        # Apply activation function
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = func.log_softmax(x, dim=1)

        return x


model = LeNet()
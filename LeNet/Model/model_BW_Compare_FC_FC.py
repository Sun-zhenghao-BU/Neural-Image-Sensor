import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define full connection layers size
        self.fc1 = nn.Linear(1 * 28 * 28, 60)
        # Define full connection layers size
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network
        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.log_softmax(x, dim=1)

        return x

model = LeNet()
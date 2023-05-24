import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define C1 layer (1 input channel, 6 output channel, kernel size is 5)
        self.conv1 = nn.Conv2d(1, 8, 5)
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define C2 layer (6 input channel, 16 output channel, kernel size is 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        # Define maxPooling2 (Filter size is 2*2)
        self.maxPool2 = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define full connection layers size
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network
        x = self.maxPool1(self.relu(self.conv1(x)))
        x = self.maxPool2(self.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        x = func.log_softmax(x, dim=1)

        return x

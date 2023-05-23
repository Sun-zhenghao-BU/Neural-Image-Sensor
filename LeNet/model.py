import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define C1 layer (1 input channel, 6 output channel, kernel size is 5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Define C2 layer (6 input channel, 16 output channel, kernel size is 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define full connection layers size
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        x = func.log_softmax(x, dim=1)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define C1 layer (1 input channel, 8 output channel, kernel size is 5)
        self.conv1 = nn.Conv2d(1, 5, 7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv2d(5, 10, 7)
        # Define a batchNorm layer
        self.bn1 = nn.BatchNorm2d(5)
        self.bn2 = nn.BatchNorm2d(10)
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define full connection layers size
        self.fc1 = nn.Linear(10 * 4 * 4, 10)

    def forward(self, x):
        x = self.maxPool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxPool(self.relu(self.bn2(self.conv2(x))))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.log_softmax(x, dim=1)

        return x



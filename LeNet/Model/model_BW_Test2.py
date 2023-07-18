import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define C2 layer (8 input channel, 16 output channel, kernel size is 7)
        self.conv2 = nn.Conv2d(8, 16, 7)
        # Define maxPooling2 (Filter size is 2*2)
        self.maxPool2 = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # Define full connection layers size
        self.fc1 = nn.Linear(16 * 4 * 4, 90)
        self.fc2 = nn.Linear(90, 63)
        self.fc3 = nn.Linear(63, 10)

    def forward(self, x):
        # Connect the input directly to the network
        x = self.maxPool1(self.relu(self.bn1(x)))
        x = self.maxPool2(self.relu(self.bn2(self.conv2(x))))

        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        x = func.log_softmax(x, dim=1)

        return x


model = LeNet()
import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # # Define maxPooling1 (Filter size is 2*2)
        # self.maxPool1 = nn.MaxPool2d(2, 2)
        # # Define ReLU activation function
        # self.relu = nn.ReLU()
        # # Define batch normalization layers
        # self.bn1 = nn.BatchNorm2d(8)
        # Define full connection layers size
        # self.fc1 = nn.Linear(8 * 14 * 14, 240)
        # self.fc2 = nn.Linear(240, 168)
        # self.fc3 = nn.Linear(168, 84)
        # self.fc4 = nn.Linear(84, 10)

        self.fc1 = nn.Linear(8 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Connect the input directly to the network
        # x = self.maxPool1(self.relu(self.bn1(x)))

        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        x = func.log_softmax(x, dim=1)

        return x


model = LeNet()


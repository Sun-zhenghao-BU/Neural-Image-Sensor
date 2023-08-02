import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import profiler


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define C2 layer (8 input channel, 16 output channel, kernel size is 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        # Define maxPooling2 (Filter size is 2*2)
        self.maxPool2 = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # Define full connection layers size
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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

# define the input data
input_data = torch.randn(1, 8, 28, 28)  # Input data demo

if torch.cuda.is_available():
    model = model.to("cuda")
    input_data = input_data.to("cuda")

    with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        # Forward Propagation
        output = model(input_data)

    # Print the result of GPU performance
    print("GPU Time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=None))
else:

    with profiler.profile(use_cuda=False, record_shapes=True) as prof:
        # Forward Propagation
        output = model(input_data)

    # Print the result of CPU performance
    print("CPU Time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))


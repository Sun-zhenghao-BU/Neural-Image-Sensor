import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import profiler


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define a batchNorm layer
        self.bn1 = nn.BatchNorm2d(8)
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(8 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxPool1(self.relu(self.bn1(x)))
        
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


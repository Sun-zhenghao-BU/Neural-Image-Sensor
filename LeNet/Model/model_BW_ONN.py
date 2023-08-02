import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import profiler


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        x = func.log_softmax(x, dim=1)

        return x


model = LeNet()

# define the input data
input_data = torch.randn(1, 1, 28, 28)  # Input data demo


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


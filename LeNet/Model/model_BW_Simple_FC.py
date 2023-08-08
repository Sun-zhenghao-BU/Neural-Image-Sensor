import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.profiler as profiler
import h5py

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define full connection layers size
        self.fc1 = nn.Linear(1 * 28 * 28, 480)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(480, 10)

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

# define the input data
file = h5py.File('../OTFData/Fashion/FashionOriginalTestSet.mat', 'r')

# Access the dataset
dataset = file['TestImages'][:]

# Get the first image
image = dataset[0]

# Convert the image to a PyTorch tensor
input_data = torch.from_numpy(image)

# Add an extra dimension for the batch size and an extra dimension for the channel
input_data = input_data.unsqueeze(0).unsqueeze(0)
input_data = input_data.type(torch.float32)

if torch.cuda.is_available():
    model = model.to("cuda")
    input_data = input_data.to("cuda")
    print("CUDA")
else:
    print("CPU")

num_iterations = 1000  # The number of times to run the model

with profiler.profile(activities=[
    # torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]) as prof:
    for _ in range(num_iterations):
        # Forward Propagation
        output = model(input_data)


# Print the result of GPU performance
print("Time:")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=None))
print("abcd")
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.profiler as profiler
import h5py


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Define C1 layer (1 input channel, 8 output channel, kernel size is 5)
        self.conv1 = nn.Conv2d(1, 8, 7)
        # Define a batchNorm layer
        self.bn1 = nn.BatchNorm2d(8)
        # Define maxPooling1 (Filter size is 2*2)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        # Define ReLU activation function
        self.relu = nn.ReLU()
        # Define full connection layers size
        self.fc1 = nn.Linear(8 * 11 * 11, 10)

    def forward(self, x):
        # Define two pooling layers and connect them to the network
        # Apply convolutional layer 1
        x = self.conv1(x)
        # Apply padding to the 8-channel output
        # Apply batch normalization
        x = self.bn1(x)
        # Apply activation function
        x = self.relu(x)
        # Apply max pooling
        x = self.maxPool1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
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

# if torch.cuda.is_available():
#     model = model.to("cuda")
#     input_data = input_data.to("cuda")
#     print("CUDA")
# else:
#     print("CPU")

num_iterations = 1000  # The number of times to run the model

with profiler.profile(activities=[
    torch.profiler.ProfilerActivity.CPU,
    # torch.profiler.ProfilerActivity.CUDA,
]) as prof:
    for _ in range(num_iterations):
        # Forward Propagation
        output = model(input_data)


# Print the result of GPU performance
print("Time:")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=None))
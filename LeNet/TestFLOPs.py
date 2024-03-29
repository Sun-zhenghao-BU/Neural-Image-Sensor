import torch
from torchsummary import summary
from thop import profile
import sys

sys.path.append("../Model")
from model_grey_Simple_FC import LeNet

# Define the model
model = LeNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_size = (1, 100, 100)

summary(model, input_size=input_size)

# Using thop lib to compute the number of FLOPs and Params
input = torch.randn(input_size).to(device)
input = input.unsqueeze(0)
MACs, params = profile(model, inputs=(input,))
# cal_ops(model, input)

print(f"Total MACs: {MACs}")
print(f"Total Params: {params}")

import torch
from torchsummary import summary
from thop import profile
from LeNet.model import LeNet


# Define the model
model = LeNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_size = (1, 28, 28)

summary(model, input_size=input_size)

# Using thop lib to compute the number of FLOPs and Params
input = torch.randn(input_size).to(device)
input = input.unsqueeze(0)  # 将输入的维度从3扩展到4
flops, params = profile(model, inputs=(input,))

print(f"Total FLOPs: {flops}")
print(f"Total Params: {params}")

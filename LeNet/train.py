import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import torch.nn.functional as func
import torch.optim as optim
from model import LeNet


def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.nelement()
            print(f'Layer: {name} | Number of parameters: {num_params}')
            total_params += num_params
    print(f'Total number of parameters: {total_params}')


Batch_size = 512
Epoch = 20
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = LeNet()
Optimizer = optim.Adam(Model.parameters())
summary(Model, (1, 28, 28))

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1037,), (0.3081,))
                   ])),
    batch_size=Batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size=Batch_size, shuffle=True)

train_loss_arr = []
test_loss_arr = []
accuracy = []
for epoch in range(1, Epoch + 1):
    Model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(Device), target.to(Device)
        Optimizer.zero_grad()
        output = Model(data)
        loss = func.nll_loss(output, target)
        loss.backward()
        Optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        elif batch_idx == 117:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, len(train_loader.dataset), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
            train_loss_arr.append(loss.item())

    Model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Device), target.to(Device)
            output = Model(data)
            test_loss += func.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_arr.append(test_loss)
    accuracy.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

fig1, ax1 = plt.subplots()
ax1.plot(accuracy, color='red', linewidth=1, linestyle='solid', label='Accuracy')
ax1.set_title('Test Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
final_accuracy = accuracy[-1]
ax1.scatter(len(accuracy) - 1, final_accuracy, color='red', label='Last Accuracy')

# Add some text annotation
ax1.annotate(f'{final_accuracy:.2f}', xy=(len(accuracy) - 1, final_accuracy),
             xytext=(len(accuracy) - 1, final_accuracy + 0.02),
             ha='center', va='bottom')

epochs = np.arange(1, len(accuracy) + 1)
ax1.set_xticks(epochs)
ax1.set_xticklabels(epochs.astype(int))

fig2, ax2 = plt.subplots()
ax2.plot(train_loss_arr, color='green', linewidth=1, linestyle='solid', label='Train Loss')
ax2.plot(test_loss_arr, color='blue', linewidth=1, linestyle='solid', label='Test Loss')
ax2.legend()
ax2.set_title('Loss Value of Train dateset and Test dataset')
ax2.set_xlabel('epoch')
ax2.set_ylabel('Loss Value')
ax2.set_xticks(epochs)

plt.show()

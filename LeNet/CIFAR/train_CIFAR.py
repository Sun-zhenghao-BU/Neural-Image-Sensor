import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as func
import torch.optim as optim
import random
import ssl
import sys
sys.path.append("../Model")
from model3 import LeNet

Batch_size = 512
Epoch = 2
Runs = 1
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

path = '../data/CIFAR'

train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(path, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)

# Select an image randomly
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]

print("Image size:", image.shape)

train_loss_runs = []
test_loss_runs = []
accuracy_runs = []

for run in range(Runs):
    print(f'Run: {run + 1}')
    train_loss_arr = []
    test_loss_arr = []
    accuracy = []
    Model = LeNet()  # Assuming your LeNet model architecture is compatible with CIFAR-10
    Model = Model.to(Device)
    Optimizer = optim.Adam(Model.parameters())

    for epoch in range(1, Epoch + 1):
        Model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(Device), target.to(Device)
            Optimizer.zero_grad()
            output = Model(data)
            loss = func.cross_entropy(output, target)
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
                test_loss += func.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_loss_arr.append(test_loss)
        accuracy.append(100. * correct / len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    train_loss_runs.append(train_loss_arr)
    test_loss_runs.append(test_loss_arr)
    accuracy_runs.append(accuracy)

avg_train_loss = np.mean(train_loss_runs, axis=0)
avg_test_loss = np.mean(test_loss_runs, axis=0)
avg_accuracy = np.mean(accuracy_runs, axis=0)

fig1, ax1 = plt.subplots()
ax1.plot(avg_accuracy, color='red', linewidth=1, linestyle='solid', label='Accuracy')
ax1.set_title('Test Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
final_accuracy = avg_accuracy[-1]
ax1.scatter(len(accuracy) - 1, final_accuracy, color='red', label='Last Accuracy')

# Add some text annotation
ax1.annotate(f'{final_accuracy:.2f}', xy=(len(accuracy) - 1, final_accuracy),
             xytext=(len(accuracy) - 1, final_accuracy + 0.02),
             ha='center', va='bottom')

epochs = np.arange(1, len(accuracy) + 1)
ax1.set_xticks(epochs)
ax1.set_xticklabels(epochs.astype(int))

fig2, ax2 = plt.subplots()
ax2.plot(avg_train_loss, color='green', linewidth=1, linestyle='solid', label='Train Loss')
ax2.plot(avg_test_loss, color='blue', linewidth=1, linestyle='solid', label='Test Loss')
ax2.legend()
ax2.set_title('Loss Value of Train dataset and Test dataset')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Value')
ax2.set_xticks(epochs)

plt.show()

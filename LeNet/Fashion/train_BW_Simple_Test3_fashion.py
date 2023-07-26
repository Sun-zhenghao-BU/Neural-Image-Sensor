from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import time
import h5py
import sys
sys.path.append("../Model")
from model_BW_Simple_CNN_FC import LeNet

# Loading .mat file
trainData = h5py.File('../OTFData/Fashion/FashionOriginalTrainSet.mat', 'r')
testData = h5py.File('../OTFData/Fashion/FashionOriginalTestSet.mat', 'r')
trainLabels = h5py.File('../OTFData/Fashion/FashionTrainLabels.mat', 'r')
testLabels = h5py.File('../OTFData/Fashion/FashionTestLabels.mat', 'r')

TrainSet = trainData['TrainImages'][:]
TestSet = testData['TestImages'][:]
TrainLabels = trainLabels['TrainLabels'][:]
TestLabels = testLabels['TestLabels'][:]

print("TrainSet shape:", TrainSet.shape)
print("TrainLabels shape:", TrainLabels.shape)
print("TestSet shape:", TestSet.shape)
print("TestLabels shape:", TestLabels.shape)

train_loss_runs = []
test_loss_runs = []
accuracy_runs = []
test_time_runs = []

Batch_size = 512
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if Device.type == 'cuda':
    print("Using CUDA for computation")
    Epoch = 20
    Runs = 5
else:
    print("Using CPU for computation")
    Epoch = 3
    Runs = 5

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(TrainSet.reshape(-1, 1, 28, 28)), torch.from_numpy(TrainLabels.squeeze())),
    batch_size=Batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(TestSet.reshape(-1, 1, 28, 28)), torch.from_numpy(TestLabels.squeeze())),
    batch_size=1,
    shuffle=True)


# Train the model
for run in range(Runs):
    print(f'Run: {run + 1}')
    train_loss_arr = []
    test_loss_arr = []
    test_time_arr = []
    accuracy = []
    Model = LeNet()
    Model = Model.to(Device)
    # Define the optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(Model.parameters(), lr=0.001)

    for epoch in range(1, Epoch + 1):
        Model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data = data.to(Device).float()
            data = data.to(Device).type(torch.float32)
            target = target.to(Device).long()
            Optimizer.zero_grad()
            output = Model(data)
            loss = criterion(output, target)
            loss.backward()
            Optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, len(train_loader.dataset), len(train_loader.dataset),
            100. * (batch_idx + 1) / len(train_loader), loss.item()))
        train_loss_arr.append(loss.item())
        Model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            test_elapsed_time = []
            for data, target in test_loader:
                data = data.to(Device).float()
                target = target.to(Device).long()

                if torch.cuda.is_available():
                    # Use CUDA event for timing if CUDA is available
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    output = Model(data)
                    pred = output.max(1, keepdim=True)[1]
                    end_time.record()

                    # Waits for everything to finish running
                    torch.cuda.synchronize()

                    test_batch_time = start_time.elapsed_time(end_time)
                else:
                    # Use Python time module for timing if CUDA is not available
                    start_time = time.time()
                    output = Model(data)
                    pred = output.max(1, keepdim=True)[1]
                    test_batch_time = time.time() - start_time

                test_elapsed_time.append(test_batch_time)

                test_loss += criterion(output, target).item() * data.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_time_arr.append(sum(test_elapsed_time))

        # test_time = time.time() - start_time
        test_loss /= len(test_loader.dataset)
        test_loss_arr.append(test_loss)
        accuracy.append(100. * correct / len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test_time_runs.append(test_time_arr)
    train_loss_runs.append(train_loss_arr)
    test_loss_runs.append(test_loss_arr)
    accuracy_runs.append(accuracy)

avg_train_loss = np.mean(train_loss_runs, axis=0)
avg_test_loss = np.mean(test_loss_runs, axis=0)
avg_accuracy = np.mean(accuracy_runs, axis=0)
avg_test_time = np.mean(test_time_runs, axis=1)
var_accuracy = np.var(accuracy_runs, axis=0)

fig1, ax1 = plt.subplots()
ax1.plot(np.arange(1, len(avg_accuracy) + 1), avg_accuracy, color='red', linewidth=1, linestyle='solid',
         label='Accuracy')
ax1.set_title('Test Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
final_accuracy = avg_accuracy[-1]
ax1.scatter(len(avg_accuracy), final_accuracy, color='red', label='Last Accuracy')

for i, acc in enumerate(avg_accuracy):
    ax1.annotate(f'{acc:.2f}', (i+1, acc), textcoords="offset points", xytext=(0, 10), ha='center')

epochs = np.arange(1, len(avg_train_loss) + 1)
ax1.set_xticks(epochs)
ax1.set_xticklabels(epochs.astype(int))

fig2, ax2 = plt.subplots()
ax2.plot(epochs, avg_train_loss, color='green', linewidth=1, linestyle='solid', label='Train Loss')
ax2.plot(epochs, avg_test_loss, color='blue', linewidth=1, linestyle='solid', label='Test Loss')
ax2.legend()
ax2.set_title('Loss Value of Train dataset and Test dataset')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss Value')
ax2.set_xticks(epochs)

fig3, ax3 = plt.subplots()
ax3.plot(np.arange(1, len(var_accuracy) + 1), var_accuracy, color='blue', linewidth=1, linestyle='solid')
ax3.set_title('Variance of Accuracy across Runs')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Variance')
ax3.set_xticks(epochs)

for epoch, var in enumerate(var_accuracy):
    ax3.annotate(f'Var: {var:.2f}', xy=(epoch+1, var), xytext=(epoch+1, var + 0.02),
                 ha='center', va='bottom')

# Display the results table
avg = np.mean(avg_test_time)

if Device.type == 'cuda':
    results_table = pd.DataFrame({
        'Run': np.arange(1, Runs + 1),
        '10000 Pics Test Time (ms)': avg_test_time
    })
    print(results_table)

    singlePicTime = avg / len(test_loader)
    print(f'{singlePicTime} ms per pic')
else:
    results_table = pd.DataFrame({
        'Run': np.arange(1, Runs + 1),
        '10000 Pics Test Time (s)': avg_test_time
    })
    print(results_table)

    singlePicTime = avg * 1000 / len(test_loader)
    print(f'{singlePicTime} ms per pic')

plt.show()
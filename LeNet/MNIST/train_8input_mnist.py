import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as func
import torch.optim as optim
from LeNet.model_8input import LeNet
import pandas as pd
import h5py
import time

# Loading .mat file
trainData = h5py.File('OTFData/MNIST/TrainSet.mat', 'r')
testData = h5py.File('OTFData/MNIST/TestSet.mat', 'r')
trainLabels = h5py.File('OTFData/MNIST/TrainLabels.mat', 'r')
testLabels = h5py.File('OTFData/MNIST/TestLabels.mat', 'r')

TrainSet = trainData['TrainSet'][:]
TestSet = testData['TestSet'][:]
TrainLabels = trainLabels['TrainLabels'][:]
TestLabels = testLabels['TestLabels'][:]

trainData.close()
testData.close()
trainLabels.close()
testLabels.close()

print("TrainSet shape:", TrainSet.shape)
print("TrainLabels shape:", TrainLabels.shape)
print("TestSet shape:", TestSet.shape)
print("TestLabels shape:", TestLabels.shape)

Batch_size = 512
Epoch = 3
Runs = 3
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(TrainSet), torch.from_numpy(TrainLabels.squeeze())),
    batch_size=Batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(TestSet), torch.from_numpy(TestLabels.squeeze())),
    batch_size=Batch_size,
    shuffle=True)

train_loss_runs = []
test_loss_runs = []
accuracy_runs = []
test_time_runs = []

for run in range(Runs):
    print(f'Run: {run + 1}')
    Model = LeNet()
    Model = Model.to(Device)
    Optimizer = optim.Adam(Model.parameters())
    train_loss_arr = []
    test_loss_arr = []
    test_time_arr = []
    accuracy = []

    for epoch in range(1, Epoch + 1):
        Model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(Device).float()
            target = target.to(Device).long()
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
        start_time = time.time()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(Device).float()
                target = target.to(Device).long()
                output = Model(data)
                test_loss += func.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_time = time.time() - start_time
        test_loss /= len(test_loader.dataset)
        test_loss_arr.append(test_loss)
        accuracy.append(100. * correct / len(test_loader.dataset))
        test_time_arr.append(test_time)
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
ax1.set_xticks(epochs - 1)
ax1.set_xticklabels(epochs.astype(int))

fig2, ax2 = plt.subplots()
ax2.plot(avg_train_loss, color='green', linewidth=1, linestyle='solid', label='Train Loss')
ax2.plot(avg_test_loss, color='blue', linewidth=1, linestyle='solid', label='Test Loss')
ax2.legend()
ax2.set_title('Loss Value of Train dateset and Test dataset')
ax2.set_xlabel('epoch')
ax2.set_ylabel('Loss Value')
ax2.set_xticks(epochs - 1)

plt.show()


# Display the results table
results_table = pd.DataFrame({
    'Run': np.arange(1, Runs + 1),
    # 'Test Accuracy': [f'{acc:.2f}%' for acc in avg_accuracy],
    # 'Test Loss': avg_test_loss,
    'Test Time (s)': avg_test_time
})

print(results_table)

avg = np.mean(avg_test_time)
print(avg)



# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import pandas as pd
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from LeNet.model_8input import LeNet
# import h5py
# import time
#
# # Loading .mat file
# trainData = h5py.File('OTFData/Fashion/FashionTrainSet.mat', 'r')
# testData = h5py.File('OTFData/Fashion/FashionTestSet.mat', 'r')
# trainLabels = h5py.File('OTFData/Fashion/FashionTrainLabels.mat', 'r')
# testLabels = h5py.File('OTFData/Fashion/FashionTestLabels.mat', 'r')
#
# TrainSet = trainData['TrainSet'][:]
# TestSet = testData['TestSet'][:]
# TrainLabels = trainLabels['TrainLabels'][:]
# TestLabels = testLabels['TestLabels'][:]
#
# print("TrainSet shape:", TrainSet.shape)
# print("TrainLabels shape:", TrainLabels.shape)
# print("TestSet shape:", TestSet.shape)
# print("TestLabels shape:", TestLabels.shape)
#
# Batch_size = 512
# Epoch = 3
# Runs = 3
# Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# train_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(torch.from_numpy(TrainSet), torch.from_numpy(TrainLabels.squeeze())),
#     batch_size=Batch_size,
#     shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(torch.from_numpy(TestSet), torch.from_numpy(TestLabels.squeeze())),
#     batch_size=Batch_size,
#     shuffle=True)
#
# train_loss_runs = []
# test_loss_runs = []
# accuracy_runs = []
# test_time_runs = []
#
# for run in range(Runs):
#     print(f'Run: {run + 1}')
#     Model = LeNet()
#     Model = Model.to(Device)
#     criterion = nn.CrossEntropyLoss()
#     Optimizer = optim.Adam(Model.parameters(), lr=0.003)
#     train_loss_arr = []
#     test_loss_arr = []
#     test_time_arr = []
#     accuracy = []
#
#     for epoch in range(1, Epoch + 1):
#         Model.train()
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data = data.to(Device).float()
#             target = target.to(Device).long()
#             Optimizer.zero_grad()
#             output = Model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             Optimizer.step()
#
#             if (batch_idx + 1) % 30 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader), loss.item()))
#
#             elif batch_idx == 117:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, len(train_loader.dataset), len(train_loader.dataset),
#                     100. * (batch_idx + 1) / len(train_loader), loss.item()))
#                 train_loss_arr.append(loss.item())
#
#         Model.eval()
#         start_time = time.time()
#         test_loss = 0
#         correct = 0
#         with torch.no_grad():
#             for data, target in test_loader:
#                 data = data.to(Device).float()
#                 target = target.to(Device).long()
#                 output = Model(data)
#                 test_loss += criterion(output, target).item() * data.size(0)
#                 pred = output.max(1, keepdim=True)[1]
#                 correct += pred.eq(target.view_as(pred)).sum().item()
#
#         test_time = time.time() - start_time
#         test_loss /= len(test_loader.dataset)
#         test_loss_arr.append(test_loss)
#         accuracy.append(100. * correct / len(test_loader.dataset))
#         test_time_arr.append(test_time)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))
#
#     test_time_runs.append(test_time_arr)
#     train_loss_runs.append(train_loss_arr)
#     test_loss_runs.append(test_loss_arr)
#     accuracy_runs.append(accuracy)
#
# avg_train_loss = np.mean(train_loss_runs, axis=0)
# avg_test_loss = np.mean(test_loss_runs, axis=0)
# avg_accuracy = np.mean(accuracy_runs, axis=0)
# avg_test_time = np.mean(test_time_runs, axis=1)
#
# fig1, ax1 = plt.subplots()
# ax1.plot(avg_accuracy, color='red', linewidth=1, linestyle='solid', label='Accuracy')
# ax1.set_title('Test Accuracy')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Accuracy')
# ax1.legend()
# final_accuracy = avg_accuracy[-1]
# ax1.scatter(len(accuracy) - 1, final_accuracy, color='red', label='Last Accuracy')
#
# # Add some text annotation
# ax1.annotate(f'{final_accuracy:.2f}', xy=(len(accuracy) - 1, final_accuracy),
#              xytext=(len(accuracy) - 1, final_accuracy + 0.02),
#              ha='center', va='bottom')
#
# epochs = np.arange(1, len(accuracy) + 1)
# ax1.set_xticks(epochs - 1)
# ax1.set_xticklabels(epochs.astype(int))
#
# fig2, ax2 = plt.subplots()
# ax2.plot(avg_train_loss, color='green', linewidth=1, linestyle='solid', label='Train Loss')
# ax2.plot(avg_test_loss, color='blue', linewidth=1, linestyle='solid', label='Test Loss')
# ax2.legend()
# ax2.set_title('Loss Value of Train dateset and Test dataset')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('Loss Value')
# ax2.set_xticks(epochs - 1)
#
# plt.show()
#
# # Display the results table
# results_table = pd.DataFrame({
#     'Run': np.arange(1, Runs + 1),
#     # 'Test Accuracy': [f'{acc:.2f}%' for acc in avg_accuracy],
#     # 'Test Loss': avg_test_loss,
#     'Test Time (s)': avg_test_time
# })
#
# print(results_table)
#
# avg = np.mean(avg_test_time)
# print(avg)
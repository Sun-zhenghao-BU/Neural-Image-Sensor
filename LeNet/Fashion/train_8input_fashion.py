import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model_8input import LeNet
import torch.nn.init as init
import h5py
import time
from fvcore.nn import FlopCountAnalysis


def train_model(train_loader, test_loader, device, Model, criterion, optimizer, num_epochs, num_Runs):
    train_loss_runs = []
    test_loss_runs = []
    accuracy_runs = []
    test_time_runs = []

    for run in range(num_Runs):
        print(f'Run: {run + 1}')
        train_loss_arr = []
        test_loss_arr = []
        test_time_arr = []
        accuracy = []
        Model.apply(reset_weights)

        for epoch in range(1, num_epochs + 1):
            Model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device).float()
                target = target.to(device).long()
                optimizer.zero_grad()
                output = Model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

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
                test_elapsed_time = []
                for data, target in test_loader:
                    data = data.to(device).float()
                    target = target.to(device).long()

                    start_time = time.time()
                    output = Model(data)
                    pred = output.max(1, keepdim=True)[1]
                    test_batch_time = time.time() - start_time
                    test_elapsed_time.append(test_batch_time)

                    test_loss += criterion(output, target).item() * data.size(0)
                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_time_arr.append(sum(test_elapsed_time))

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

    return avg_train_loss, avg_test_loss, avg_accuracy, avg_test_time


def plot_results(avg_accuracy, avg_train_loss, avg_test_loss):
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(avg_accuracy) + 1), avg_accuracy, color='red', linewidth=1, linestyle='solid',
             label='Accuracy')
    ax1.set_title('Test Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    final_accuracy = avg_accuracy[-1]
    ax1.scatter(len(avg_accuracy), final_accuracy, color='red', label='Last Accuracy')

    ax1.annotate(f'{final_accuracy:.2f}', xy=(len(avg_accuracy), final_accuracy),
                 xytext=(len(avg_accuracy), final_accuracy + 0.02),
                 ha='center', va='bottom')

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

    plt.show()


# Display the results table
def plot_time(avg_test_time, Runs):
    results_table = pd.DataFrame({
        'Run': np.arange(1, Runs + 1),
        'Test Time (s)': avg_test_time
    })

    print(results_table)
    avg = np.mean(avg_test_time)
    print(avg)


def cal_ops(model, input):
    f1 = FlopCountAnalysis(model, input)
    print(f1.total())

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def main():
    # Loading .mat file
    trainData = h5py.File('../OTFData/Fashion/FashionTrainSet.mat', 'r')
    testData = h5py.File('../OTFData/Fashion/FashionTestSet.mat', 'r')
    trainLabels = h5py.File('../OTFData/Fashion/FashionTrainLabels.mat', 'r')
    testLabels = h5py.File('../OTFData/Fashion/FashionTestLabels.mat', 'r')

    TrainSet = trainData['TrainSet'][:]
    TestSet = testData['TestSet'][:]
    TrainLabels = trainLabels['TrainLabels'][:]
    TestLabels = testLabels['TestLabels'][:]

    print("TrainSet shape:", TrainSet.shape)
    print("TrainLabels shape:", TrainLabels.shape)
    print("TestSet shape:", TestSet.shape)
    print("TestLabels shape:", TestLabels.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Batch_size = 512
    Epoch = 20
    Runs = 10
    Model = LeNet()
    Model = Model.to(device)
    criterion = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(Model.parameters(), lr=0.001)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(TrainSet), torch.from_numpy(TrainLabels.squeeze())),
        batch_size=Batch_size,
        shuffle=True)

    test_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(TestSet), torch.from_numpy(TestLabels.squeeze())),
        batch_size=Batch_size,
        shuffle=True)

    avg_train_loss, avg_test_loss, avg_accuracy, avg_test_time = train_model(
        train_loader, test_loader, device, Model, criterion, Optimizer, Epoch, Runs)

    plot_results(avg_accuracy, avg_train_loss, avg_test_loss)
    plot_time(avg_test_time, Runs)


if __name__ == '__main__':
    main()

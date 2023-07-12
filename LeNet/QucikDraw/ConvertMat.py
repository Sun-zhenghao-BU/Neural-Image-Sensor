import numpy as np
from scipy.io import savemat

# Load the .npy files
train_data = np.load('../data/QuickDraw/train_data.npy')
train_labels = np.load('../data/QuickDraw/train_labels.npy')
test_data = np.load('../data/QuickDraw/test_data.npy')
test_labels = np.load('../data/QuickDraw/test_labels.npy')

# Convert the data to MATLAB-readable format
train_data_mat = {'TrainData': train_data}
train_labels_mat = {'TrainLabels': train_labels}
test_data_mat = {'TestData': test_data}
test_labels_mat = {'TestLabels': test_labels}

# Save as .mat files
savemat('../OTFData/QuickDraw/TrainData.mat', train_data_mat)
savemat('../OTFData/QuickDraw/TrainLabels.mat', train_labels_mat)
savemat('../OTFData/QuickDraw/TestData.mat', test_data_mat)
savemat('../OTFData/QuickDraw/TestLabels.mat', test_labels_mat)

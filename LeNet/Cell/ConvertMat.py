import numpy as np
from scipy.io import savemat

# Load the npz file
data = np.load('../data/Cell/EBI_Cells.npz')

# Extract required datasets
TrainSet = data['train_data_grey'].reshape(-1, 100, 100)
TrainLabels = data['train_labels']
TestSet = data['test_data_grey'].reshape(-1, 100, 100)
TestLabels = data['test_labels']

# Convert the data to MATLAB-readable format and save
savemat('../OTFData/Cell/TrainSet.mat', {'TrainSet': TrainSet})
savemat('../OTFData/Cell/TrainLabels.mat', {'TrainLabels': TrainLabels})
savemat('../OTFData/Cell/TestSet.mat', {'TestSet': TestSet})
savemat('../OTFData/Cell/TestLabels.mat', {'TestLabels': TestLabels})

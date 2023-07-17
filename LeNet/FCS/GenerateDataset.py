from FlowCytometryTools import FCMeasurement

sample = FCMeasurement(ID='Sample', datafile='../data/FlowRepository_FR-FCM-Z4M5_files/2A_R1.fcs')
data = sample.data

print(data.head())
import numpy as np

# 加载npz文件
data = np.load('../data/Cell/EBI_Cells.npz', allow_pickle=True)


# 列出文件中所有的数组
for name in data.files:
    array_content = data[name]
    print(f"Array Name: {name}")
    print(f"Shape: {array_content.shape}")
    print(f"Data Type: {array_content.dtype}")
    print("-------")

# train_data_bin = data['train_data_bin']
# train_data_grey = data['train_data_grey']
# train_labels = data['train_labels']
# train_files = data['train_files']
# test_data_bin = data['test_data_bin']
# test_data_grey = data['test_data_grey']
# test_labels = data['test_labels']
# test_files = data['test_files']


# 通过键名访问特定的数组
# array_a = data['arr_0']
# 这里 'arr_0' 是一个示例，你应该使用实际的键名

# 如果你知道文件中的数组名称，也可以这样做：
# array_a = data['array_name']  # 替换 'array_name' 为实际的名称

# print(array_a)

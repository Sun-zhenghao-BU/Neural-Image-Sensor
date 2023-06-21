import os
import json
import random

# 定义 JSON 文件夹路径和输出图片文件夹路径
json_folder = "../data/data/"
output_folder = "../data/png_files/"

# 定义抽样数量和训练集、测试集的比例
sample_size = 12000
train_size = 10000
test_size = 2000

# 存储抽样结果的列表
sampled_examples = []

# 遍历 JSON 文件夹中的所有文件
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        # 构建 JSON 文件的完整路径
        json_path = os.path.join(json_folder, filename)

        # 读取 JSON 文件数据
        with open(json_path) as f:
            settings = json.load(f)

        # 随机抽取指定数量的例子，并为每个例子分配标签
        random_examples = random.sample(settings, sample_size)
        labeled_examples = [(example, filename.split(".")[0]) for example in random_examples]

        # 将抽样结果添加到总体样本列表中
        sampled_examples.extend(labeled_examples)

# 打印抽样结果的数量
print(f"总共抽取了 {len(sampled_examples)} 个例子")

# 随机打乱样本顺序
random.shuffle(sampled_examples)

# 构建训练集和测试集，并分别提取样本和标签
train_set = sampled_examples[:train_size]
test_set = sampled_examples[train_size:train_size+test_size]
train_data, train_labels = zip(*train_set)
test_data, test_labels = zip(*test_set)

# 打印训练集和测试集的数量
print(f"训练集数量：{len(train_data)}")
print(f"测试集数量：{len(test_data)}")

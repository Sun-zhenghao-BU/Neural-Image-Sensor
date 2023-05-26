import os
import random
import shutil
from torchvision import datasets

# 设置随机种子
random.seed(1234)

# 加载 MNIST 数据集
mnist_dataset = datasets.MNIST('data', train=True, download=True)
mnist_list = list(mnist_dataset)
# 随机选择 100 张图片
random_samples = random.sample(mnist_list, 100)

# 创建保存图片的文件夹
output_folder = 'random_images'
os.makedirs(output_folder, exist_ok=True)

# 将图片保存到文件夹中
for i, (image, label) in enumerate(random_samples):
    image_path = os.path.join(output_folder, f'image_{i+1}.png')
    image.save(image_path)

print("Random images saved to", output_folder)
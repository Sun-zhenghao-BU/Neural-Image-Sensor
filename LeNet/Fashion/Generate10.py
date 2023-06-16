import os
import random
from torchvision import datasets

# Set Random Seed
random.seed(1234)

# Load MNIST dataset
mnist_dataset = datasets.FashionMNIST('data', train=True, download=True)
mnist_list = list(mnist_dataset)
# Random Select 10 pictures from the test dataset
random_samples = random.sample(mnist_list, 10)

# Create the folder to store the random images
output_folder = 'random_images'
os.makedirs(output_folder, exist_ok=True)

# Store these pictures to the folder
for i, (image, label) in enumerate(random_samples):
    image_path = os.path.join(output_folder, f'image_{i+1}.png')
    image.save(image_path)

print("Random images saved to", output_folder)
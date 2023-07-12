import os
import json
import random
import numpy as np

# Define the path where the JSON files are located
json_folder = "../data/data/Dataset/"
output_folder = "../data/QuickDraw/"

# Define the size and proportion of the training set and test set in the dataset
train_size_per_file = 10000
test_size_per_file = 2000

# Store the list of sampling results
train_examples = []
test_examples = []

# Traverse all the JSON files
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        # Create the full path of the JSON file
        json_path = os.path.join(json_folder, filename)

        # Read the data from the JSON file
        with open(json_path) as f:
            settings = json.load(f)

        # Randomly select a given number of datasets and assign labels to them
        random_train_examples = random.sample(settings, train_size_per_file)
        random_test_examples = random.sample(settings, test_size_per_file)

        train_labeled_examples = [(example, filename.split(".")[0]) for example in random_train_examples]
        test_labeled_examples = [(example, filename.split(".")[0]) for example in random_test_examples]

        # Add these samples to the example lists
        train_examples.extend(train_labeled_examples)
        test_examples.extend(test_labeled_examples)

print(f"Total examples: {len(train_examples) + len(test_examples)}")

random.shuffle(train_examples)
random.shuffle(test_examples)

# Create the training and test datasets
train_data, train_labels = zip(*train_examples)
test_data, test_labels = zip(*test_examples)

# Define the label mapping
label_mapping = {
    'clock': 0,
    'chair': 1,
    'computer': 2,
    'eyeglasses': 3,
    'tent': 4,
    'snowflake': 5,
    'pants': 6,
    'hurricane': 7,
    'flower': 8,
    'crown': 9
}

# Convert the labels to numeric encoding
train_labels_encoded = np.array([label_mapping[label] for label in train_labels], dtype=np.int64)
test_labels_encoded = np.array([label_mapping[label] for label in test_labels], dtype=np.int64)

# Save PNG images and pixel values
train_data_pixels = []
test_data_pixels = []

# Convert pixel values to NumPy arrays
train_data_pixels = np.array(train_data_pixels, dtype=np.uint8)
test_data_pixels = np.array(test_data_pixels, dtype=np.uint8)

# Save as .npy files
np.save(os.path.join(output_folder, "train_data.npy"), train_data_pixels)
np.save(os.path.join(output_folder, "test_data.npy"), test_data_pixels)
np.save(os.path.join(output_folder, "train_labels.npy"), train_labels_encoded)
np.save(os.path.join(output_folder, "test_labels.npy"), test_labels_encoded)

print("Data saved successfully!")


# # Traverse the drawing data in the training set and save PNG images and pixel values
# for i, drawing in enumerate(train_data):
#     plt.figure()
#     for stroke in drawing['drawing']:
#         x = stroke[0]
#         y = stroke[1]
#         plt.plot(x, y, 'k')
#
#     ax = plt.gca()
#     ax.xaxis.set_ticks_position('top')
#     ax.invert_yaxis()
#     plt.axis('off')
#
#     # Set the image size to 28x28 pixels
#     fig = plt.gcf()
#     fig.set_size_inches(0.28, 0.28)
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#
#     # Create the output subfolder if it doesn't exist
#     output_subfolder = os.path.join(output_folder, "train")
#     os.makedirs(output_subfolder, exist_ok=True)
#
#     # Save the PNG image
#     png_path = os.path.join(output_subfolder, f"{i}.png")
#     plt.savefig(png_path, dpi=100)
#
#     # Read the PNG image and convert it to pixel values
#     image = Image.open(png_path).convert("L")  # Convert to grayscale
#     pixels = np.array(image)
#     train_data_pixels.append(pixels)
#
#     # Print the image size
#     width, height = image.size
#     print(f"Image {png_path} size: {width} x {height}")
#
#     plt.close()

# # Traverse the drawing data in the test set and save PNG images and pixel values
# for i, drawing in enumerate(test_data):
#     plt.figure()
#     for stroke in drawing['drawing']:
#         x = stroke[0]
#         y = stroke[1]
#         plt.plot(x, y, 'k')
#
#     ax = plt.gca()
#     ax.xaxis.set_ticks_position('top')
#     ax.invert_yaxis()
#     plt.axis('off')
#
#     # Set the image size to 28x28 pixels
#     fig = plt.gcf()
#     fig.set_size_inches(0.28, 0.28)
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#
#     # Create the output subfolder if it doesn't exist
#     output_subfolder = os.path.join(output_folder, "test")
#     os.makedirs(output_subfolder, exist_ok=True)
#
#     # Save the PNG image
#     png_path = os.path.join(output_subfolder, f"{i}.png")
#     plt.savefig(png_path, dpi=100)
#
#     # Read the PNG image and convert it to pixel values
#     image = Image.open(png_path).convert("L")  # Convert to grayscale
#     pixels = np.array(image)
#     test_data_pixels.append(pixels)
#
#     # Print the image size
#     width, height = image.size
#     print(f"Image {png_path} size: {width} x {height}")
#
#     plt.close()
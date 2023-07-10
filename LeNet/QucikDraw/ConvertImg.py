import os
import json
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image

# Define input and output path
json_folder = "../data/data/"
output_folder = "../data/png_files/"

# Traverse all the folder under this path
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):

        json_path = os.path.join(json_folder, filename)

        # Read Json file
        with open(json_path) as f:
            settings = json.load(f)

        # Create output folder
        output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(output_subfolder, exist_ok=True)

        # Traverse drawing for each data
        for j in range(len(settings)):
            plt.figure()
            for i in range(len(settings[j]['drawing'])):
                x = settings[j]['drawing'][i][0]
                y = settings[j]['drawing'][i][1]
                f = interp1d(x, y, kind="linear")
                plt.plot(x, y, 'k')

            ax = plt.gca()
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            plt.axis('off')

            # Open and resize the image to 28*28
            fig = plt.gcf()
            fig.set_size_inches(0.28, 0.28)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(os.path.join(output_subfolder, f"{j}.png"), dpi=100)
            plt.close()

            image_path = os.path.join(output_subfolder, f"{j}.png")
            image = Image.open(image_path)
            width, height = image.size
            print(f"Image {image_path} size is：{width} x {height}")

print("Complete Transferring")

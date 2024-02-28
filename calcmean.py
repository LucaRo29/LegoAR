import os
import numpy as np
import cv2

folder_path = "../data/legoarimage"

# Define lists to store channel means and stds
means = []
stds = []
counter = 0
# Iterate over all files in folder and subfolders
for root, dirs, files in os.walk(folder_path):
    print(dirs)
    for file in files:
        # Check if file is an image file
        if file.endswith((".jpg", ".jpeg", ".png")):
            # Read image file
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            # Compute mean and std for each channel
            channel_means = np.mean(img, axis=(0,1))
            channel_stds = np.std(img, axis=(0,1))
            # Append channel means and stds to lists
            means.append(channel_means)
            stds.append(channel_stds)

# Compute average channel means and stds across all images
mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)

print("Mean: ", mean)
print("Std: ", std)

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import ast


import glob
import os
from tqdm import tqdm
import cv2
from skimage.feature import hog

data_files = glob.glob("data/processed/data_batch_*.csv")

for file_path in data_files:
    df = pd.read_csv(file_path)

    hog_descriptors = []
    for i in tqdm(range(df.shape[0])):
        image_data = df.iloc[i]["data"]
        image_data = ast.literal_eval(image_data)
        image = np.reshape(image_data, (32, 32, 3), order='F')
        image = np.transpose(image, (1, 0, 2))
        image = image.astype('uint8')
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hog_descriptor = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=False)
        hog_descriptors.append(hog_descriptor.tolist())

    df["hog_descriptor"] = hog_descriptors

    new_file_path = file_path.replace("processed", "interim/hog")

    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    df.to_csv(new_file_path, index=False)

#%%

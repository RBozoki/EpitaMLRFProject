import glob
import os
import pandas as pd
import ast
import cv2
import numpy as np
from tqdm import tqdm

fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)

data_files = glob.glob("data/processed/data_batch_*.csv")

for file_path in data_files:
    df = pd.read_csv(file_path)

    brief_descriptors = []
    for i in tqdm(range(df.shape[0])):
        image_data = df.iloc[i]["data"]
        image_data = ast.literal_eval(image_data)
        image = np.reshape(image_data, (32, 32, 3), order='F')
        image = np.transpose(image, (1, 0, 2))
        image = image.astype('uint8')
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints = fast.detect(image_gray, None)

        keypoints, descriptor = brief.compute(image_gray, keypoints)

        if descriptor is not None:
            descriptor = descriptor.flatten()
        else:
            descriptor = np.zeros(brief.descriptorSize(), dtype=np.float32)

        brief_descriptors.append(descriptor.tolist())

    df["brief_descriptor"] = brief_descriptors

    new_file_path = file_path.replace("processed", "interim/brief")

    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    df.to_csv(new_file_path, index=False)
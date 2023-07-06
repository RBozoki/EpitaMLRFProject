import numpy as np
import cv2
from skimage import feature
import matplotlib.pyplot as plt
import ast
import pandas as pd

df = pd.read_csv("data/processed/data_batch_1.csv")

first_image = df.iloc[0]["data"]
first_image = ast.literal_eval(first_image)
image = np.reshape(first_image, (32, 32, 3), order='F')
image = image.astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lbp = feature.local_binary_pattern(image, P=8, R=1, method="uniform")

plt.figure(figsize=(8, 8))
plt.imshow(lbp, cmap='gray')
plt.axis('off')
plt.show()
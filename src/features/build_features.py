import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import ast

df = pd.read_csv("../../data/processed/data_batch_1.csv")

first_image = df.iloc[0]["data"]
first_image = ast.literal_eval(first_image)
print(len(first_image))
image = np.reshape(first_image, (32, 32, 3), order='F')
image = np.transpose(image, (1, 0, 2))

# Afficher l'image
plt.imshow(image)
plt.show()

import cv2

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialiser le descripteur HOG.
win_size = (32, 32)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
deriv_aperture = 1
win_sigma = -1.
histogram_norm_type = 0
l2_hys_threshold = 0.2
gamma_correction = 1
nlevels = 64
signed_gradients = True

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels, signed_gradients)

# Calculer le descripteur HOG.
hog_descriptor = hog.compute(image_gray)

print(hog_descriptor)

#%%

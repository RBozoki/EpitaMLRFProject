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
from skimage.feature import hog
from skimage import exposure

image = image.astype('uint8')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculer le descripteur HOG et obtenir une visualisation
hog_descriptor, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True)

# Augmenter le contraste de l'image HOG en utilisant l'égalisation d'histogramme
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Afficher l'image HOG
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.show()

#%%

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

from tqdm import tqdm

# Initialisez une liste vide pour stocker les descripteurs HOG
hog_descriptors = []

# Bouclez sur toutes les images du DataFrame
for i in tqdm(range(df.shape[0])):
    image_data = df.iloc[i]["data"]
    image_data = ast.literal_eval(image_data)
    image = np.reshape(image_data, (32, 32, 3), order='F')
    image = np.transpose(image, (1, 0, 2))
    image = image.astype('uint8')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculez le descripteur HOG pour l'image
    hog_descriptor = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualize=False)

    # Ajoutez le descripteur HOG à la liste
    hog_descriptors.append(hog_descriptor)

# Ajoutez les descripteurs HOG en tant que nouvelle colonne dans le DataFrame
df["hog_descriptor"] = hog_descriptors

# Afficher les premières lignes du DataFrame pour vérifier
print(df.head())
#%%

import glob
import os
import pandas as pd
import ast
import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm

# LBP settings
numPoints = 24
radius = 8

data_files = glob.glob("data/processed/data_batch_*.csv")

# Bouclez sur tous les fichiers trouvés
for file_path in data_files:
    df = pd.read_csv(file_path)

    lbp_descriptors = []
    for i in tqdm(range(df.shape[0])):
        image_data = df.iloc[i]["data"]
        image_data = ast.literal_eval(image_data)
        image = np.reshape(image_data, (32, 32, 3), order='F')
        image = np.transpose(image, (1, 0, 2))
        image = image.astype('uint8')
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extraction de descripteurs LBP
        lbp = feature.local_binary_pattern(image_gray, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        lbp_descriptors.append(hist.tolist())

    df["lbp_descriptor"] = lbp_descriptors

    # Créer un nouveau chemin de fichier pour sauvegarder les données prétraitées
    new_file_path = file_path.replace("processed", "interim/lbp")

    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Enregistrez le DataFrame dans le nouveau fichier
    df.to_csv(new_file_path, index=False)

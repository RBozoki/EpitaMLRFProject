import glob
import os
import pandas as pd
import ast
import cv2
import numpy as np
from tqdm import tqdm

# Initialisation de SIFT
sift = cv2.xfeatures2d.SIFT_create()

data_files = glob.glob("data/processed/data_batch_*.csv")

# Bouclez sur tous les fichiers trouvés
for file_path in data_files:
    df = pd.read_csv(file_path)

    sift_descriptors = []
    for i in tqdm(range(df.shape[0])):
        image_data = df.iloc[i]["data"]
        image_data = ast.literal_eval(image_data)
        image = np.reshape(image_data, (32, 32, 3), order='F')
        image = np.transpose(image, (1, 0, 2))
        image = image.astype('uint8')
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detection de points clés et extraction de descripteurs avec SIFT
        keypoints, descriptor = sift.detectAndCompute(image_gray, None)

        if descriptor is not None:
            # Les descripteurs sont renvoyés comme un tableau 2D,
            # nous devons donc les aplatir en un tableau 1D
            descriptor = descriptor.flatten()
        else:
            # Si aucun point clé n'est détecté, le descripteur sera None,
            # donc nous devons créer un descripteur de zéros
            descriptor = np.zeros(sift.descriptorSize(), dtype=np.float32)

        sift_descriptors.append(descriptor.tolist())

    df["sift_descriptor"] = sift_descriptors

    # Créer un nouveau chemin de fichier pour sauvegarder les données prétraitées
    new_file_path = file_path.replace("processed", "interim/sift")

    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Enregistrez le DataFrame dans le nouveau fichier
    df.to_csv(new_file_path, index=False)
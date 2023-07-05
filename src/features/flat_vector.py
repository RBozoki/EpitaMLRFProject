import glob
import os
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

data_files = glob.glob("data/processed/data_batch_*.csv")

# Bouclez sur tous les fichiers trouvés
for file_path in data_files:
    df = pd.read_csv(file_path)

    pixel_data = []
    for i in tqdm(range(df.shape[0])):
        image_data = df.iloc[i]["data"]
        image_data = ast.literal_eval(image_data)
        pixel_data.append(image_data)

    # Création d'un nouveau DataFrame pour les pixels
    df_pixels = pd.DataFrame(pixel_data, columns=[f'pixel_{i}' for i in range(len(pixel_data[0]))])

    # Concaténation du DataFrame original avec le nouveau DataFrame de pixels
    df = pd.concat([df, df_pixels], axis=1)

    # Créer un nouveau chemin de fichier pour sauvegarder les données prétraitées
    new_file_path = file_path.replace("processed", "interim/pixels")

    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Enregistrez le DataFrame dans le nouveau fichier
    df.to_csv(new_file_path, index=False)
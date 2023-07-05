import ast
import glob
import xgboost as xgb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix

matplotlib.use('TkAgg')

accuracies = {}

# Les sous-dossiers pour lesquels nous allons former et tester le modèle
subfolders = ['hog', 'brief', 'flat']

for subfolder in subfolders:
    print(f"Training and testing model for {subfolder} data...")

    # Trouver tous les fichiers correspondant à 'data_batch_*.csv' sauf 'data_batch_test.csv'
    data_files = [f for f in glob.glob(f"data/interim/{subfolder}/data_batch_*.csv") if "test" not in f]

    # initialiser des listes vides pour stocker les descripteurs et les étiquettes
    descriptors = []
    labels = []

    # boucler sur tous les fichiers trouvés
    for file_path in data_files:
        df = pd.read_csv(file_path)
        for i in range(df.shape[0]):
            # Sélectionner le bon descripteur en fonction du type de données
            if subfolder == 'hog':
                descriptor = ast.literal_eval(df.iloc[i]["hog_descriptor"])
            elif subfolder == 'brief':
                descriptor = ast.literal_eval(df.iloc[i]["brief_descriptor"])
            elif subfolder == 'flat':
                descriptor = df.filter(regex=("pixel_.*")).iloc[i].values.tolist()
            descriptors.append(descriptor)
            labels.append(df.iloc[i]["labels"])

    # Convertir les listes en np.array
    descriptors = np.array(descriptors)
    labels = np.array(labels)

    # Initialiser et entrainer le modèle XGBoost
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb_model.fit(descriptors, labels)

    # Charger le fichier data_batch_test.csv
    df_test = pd.read_csv(f"data/interim/{subfolder}/data_batch_test.csv")
    descriptors_test = []
    labels_test = []
    for i in range(df_test.shape[0]):
        # Sélectionner le bon descripteur en fonction du type de données
        if subfolder == 'hog':
            descriptor = ast.literal_eval(df_test.iloc[i]["hog_descriptor"])
        elif subfolder == 'brief':
            descriptor = ast.literal_eval(df_test.iloc[i]["brief_descriptor"])
        elif subfolder == 'flat':
            descriptor = df_test.filter(regex=("pixel_.*")).iloc[i].values.tolist()
        descriptors_test.append(descriptor)
        labels_test.append(df_test.iloc[i]["labels"])

    descriptors_test = np.array(descriptors_test)
    labels_test = np.array(labels_test)

    # Prédire sur l'ensemble de test
    y_pred = xgb_model.predict(descriptors_test)

    # Calculer la précision
    accuracy = accuracy_score(labels_test, y_pred)
    accuracies[subfolder] = accuracy

print("\nPrécisions des modèles XGBoost :\n")
print("{:<10} {:<10}".format('Data', 'Accuracy'))
for k, v in accuracies.items():
    print("{:<10} {:<10.2f}".format(k, v))

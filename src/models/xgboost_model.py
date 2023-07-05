import ast
import glob
import xgboost as xgb

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

accuracies = {}
f1_scores = {}

# Les sous-dossiers pour lesquels nous allons former et tester le modèle
#subfolders = ['hog', 'brief', 'flat']
#subfolders = ['hog', 'brief', 'sift']
subfolders = ['hog', 'brief']

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
            elif subfolder == 'sift':
                descriptor = ast.literal_eval(df.iloc[i]["sift_descriptor"])
            elif subfolder == 'flat':
                descriptor = df.filter(regex=("pixel_.*")).iloc[i].values.tolist()
            descriptors.append(descriptor)
            labels.append(df.iloc[i]["labels"])

    # Normaliser les longueurs des descripteurs
    max_len = max(len(d) for d in descriptors)
    descriptors = [d + [0]*(max_len-len(d)) for d in descriptors]

# Convertir les listes en np.array
    descriptors = np.array(descriptors)
    labels = np.array(labels)

    # Initialiser et entrainer le modèle XGBoost
    print("Training...")
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
        elif subfolder == 'sift':
            descriptor = ast.literal_eval(df_test.iloc[i]["sift_descriptor"])
        elif subfolder == 'flat':
            descriptor = df_test.filter(regex=("pixel_.*")).iloc[i].values.tolist()
        descriptors_test.append(descriptor)
        labels_test.append(df_test.iloc[i]["labels"])

    descriptors_test = np.array(descriptors_test)
    labels_test = np.array(labels_test)

    # Prédire sur l'ensemble de test
    print("Testing...")
    y_pred = xgb_model.predict(descriptors_test)

    # Calculer la précision
    accuracy = accuracy_score(labels_test, y_pred)
    f1 = f1_score(labels_test, y_pred, average='weighted')  # Calculer le score F1
    accuracies[subfolder] = accuracy
    f1_scores[subfolder] = f1

print("\nPrécisions et scores F1 des modèles XGBoost :\n")
print("{:<10} {:<10} {:<10}".format('Data', 'Accuracy', 'F1 Score'))  # Ajouter une colonne pour le score F1
for k in accuracies.keys():
    print("{:<10} {:<10.2f} {:<10.2f}".format(k, accuracies[k], f1_scores[k]))
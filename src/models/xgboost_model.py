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

# Les sous-dossiers pour lesquels nous allons former et tester le modèle
subfolders = ['hog', 'brief', 'flat']

for subfolder in subfolders:
    print(f"Training and testing model for {subfolder} data...")

    # Trouver tous les fichiers correspondant à 'data_batch_*.csv' sauf 'data_batch_test.csv'
    data_files = [f for f in glob.glob(f"../../data/interim/{subfolder}/data_batch_*.csv") if "test" not in f]

    # initialiser des listes vides pour stocker les descripteurs et les étiquettes
    descriptors = []
    labels = []

    # boucler sur tous les fichiers trouvés
    for file_path in data_files:
        df = pd.read_csv(file_path)
        for i in range(df.shape[0]):
            descriptor = ast.literal_eval(df.iloc[i]["descriptor"])
            descriptors.append(descriptor)
            labels.append(df.iloc[i]["labels"])

    # Convertir les listes en np.array
    descriptors = np.array(descriptors)
    labels = np.array(labels)

    # Initialiser et entrainer le modèle XGBoost
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb_model.fit(descriptors, labels)

    # Charger le fichier data_batch_test.csv
    df_test = pd.read_csv(f"../../data/interim/{subfolder}/data_batch_test.csv")
    descriptors_test = []
    labels_test = []
    for i in range(df_test.shape[0]):
        descriptor = ast.literal_eval(df_test.iloc[i]["descriptor"])
        descriptors_test.append(descriptor)
        labels_test.append(df_test.iloc[i]["labels"])

    descriptors_test = np.array(descriptors_test)
    labels_test = np.array(labels_test)

    # Prédire sur l'ensemble de test
    y_pred = xgb_model.predict(descriptors_test)

    # Calculer la précision
    accuracy = accuracy_score(labels_test, y_pred)
    print(f"La précision du modèle XGBoost pour {subfolder} data est : {accuracy}")

    cm = confusion_matrix(labels_test, y_pred)

    # Convertir la matrice de confusion en pourcentages
    cm_percentage = cm / np.sum(cm, axis=1)[:, np.newaxis]

    # Afficher la matrice de confusion en pourcentages
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues')
    plt.title(f'Matrice de confusion en pourcentage pour {subfolder} data')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.show()

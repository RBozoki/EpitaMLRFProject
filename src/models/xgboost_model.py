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

# Trouver tous les fichiers correspondant à 'data_batch_*.csv' sauf 'data_batch_test.csv'
data_files = [f for f in glob.glob("../../data/interim/data_batch_*.csv") if "test" not in f]

# initialiser des listes vides pour stocker les descripteurs HOG et les étiquettes
hog_descriptors = []
labels = []

# boucler sur tous les fichiers trouvés
for file_path in data_files:
    df = pd.read_csv(file_path)
    for i in range(df.shape[0]):
        hog_descriptor = ast.literal_eval(df.iloc[i]["hog_descriptor"])
        hog_descriptors.append(hog_descriptor)
        labels.append(df.iloc[i]["labels"])

# Convertir les listes en np.array
hog_descriptors = np.array(hog_descriptors)
labels = np.array(labels)

# Initialiser et entrainer le modèle XGBoost
xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
xgb_model.fit(hog_descriptors, labels)

# Charger le fichier data_batch_test.csv
df_test = pd.read_csv("../../data/interim/data_batch_test.csv")
hog_descriptors_test = []
labels_test = []
for i in range(df_test.shape[0]):
    hog_descriptor = ast.literal_eval(df_test.iloc[i]["hog_descriptor"])
    hog_descriptors_test.append(hog_descriptor)
    labels_test.append(df_test.iloc[i]["labels"])

hog_descriptors_test = np.array(hog_descriptors_test)
labels_test = np.array(labels_test)

# Prédire sur l'ensemble de test
y_pred = xgb_model.predict(hog_descriptors_test)

# Calculer la précision
accuracy = accuracy_score(labels_test, y_pred)
print(f"La précision du modèle XGBoost est : {accuracy}")

cm = confusion_matrix(labels_test, y_pred)

# Convertir la matrice de confusion en pourcentages
cm_percentage = cm / np.sum(cm, axis=1)[:, np.newaxis]

# Afficher la matrice de confusion en pourcentages
plt.figure(figsize=(10, 10))
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues')
plt.title('Matrice de confusion en pourcentage')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.show()

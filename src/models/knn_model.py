import ast
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Initialiser et entrainer le modèle kNN
knn = KNeighborsClassifier(n_neighbors=80)
knn.fit(hog_descriptors, labels)

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
y_pred = knn.predict(hog_descriptors_test)

# Calculer la précision
accuracy = accuracy_score(labels_test, y_pred)
print(f"La précision du modèle kNN est : {accuracy}")


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

print("hello")
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
#%%


# Liste des valeurs de k pour lesquelles nous voulons tester la performance
k_values = [1, 3, 5, 7, 10, 15, 20, 30, 40, 80, 200]
accuracy_scores = [0.2323, 0.2375, 0.2647, 0.272, 0.2872, 0.2999, 0.3051, 0.3089, 0.3132, 0.315, 0.3009]

# Tracer l'accuracy en fonction de k
plt.plot(k_values, accuracy_scores)
plt.ylim([0, 0.5])
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy en fonction de k pour le modèle k-NN')
plt.show()
#%%

import ast
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

accuracies = {}
f1_scores = {}

subfolders = ['hog', 'lbp']

for subfolder in subfolders:
    print(f"Training and testing model for {subfolder} data...")

    data_files = [f for f in glob.glob(f"data/interim/{subfolder}/data_batch_*.csv") if "test" not in f]

    descriptors = []
    labels = []

    for file_path in data_files:
        df = pd.read_csv(file_path)
        for i in range(df.shape[0]):
            if subfolder == 'hog':
                descriptor = ast.literal_eval(df.iloc[i]["hog_descriptor"])
            elif subfolder == 'lbp':
                descriptor = ast.literal_eval(df.iloc[i]["lbp_descriptor"])
            descriptors.append(descriptor)
            labels.append(df.iloc[i]["labels"])

    max_len = max(len(d) for d in descriptors)
    descriptors = [d + [0]*(max_len-len(d)) for d in descriptors]

    descriptors = np.array(descriptors)
    labels = np.array(labels)

    print("Training...")
    log_reg = LogisticRegression(multi_class='multinomial', max_iter=10000)
    log_reg.fit(descriptors, labels)

    df_test = pd.read_csv(f"data/interim/{subfolder}/data_batch_test.csv")
    descriptors_test = []
    labels_test = []
    for i in range(df_test.shape[0]):
        if subfolder == 'hog':
            descriptor = ast.literal_eval(df_test.iloc[i]["hog_descriptor"])
        elif subfolder == 'lbp':
            descriptor = ast.literal_eval(df_test.iloc[i]["lbp_descriptor"])
        descriptors_test.append(descriptor)
        labels_test.append(df_test.iloc[i]["labels"])

    descriptors_test = np.array(descriptors_test)
    labels_test = np.array(labels_test)

    print("Testing...")
    y_pred = log_reg.predict(descriptors_test)

    accuracy = accuracy_score(labels_test, y_pred)
    f1 = f1_score(labels_test, y_pred, average='weighted')
    accuracies[subfolder] = accuracy
    f1_scores[subfolder] = f1

print("\nPrécisions et scores F1 des modèles de régression logistique :\n")
print("{:<10} {:<10} {:<10}".format('Data', 'Accuracy', 'F1 Score'))
for k in accuracies.keys():
    print("{:<10} {:<10.2f} {:<10.2f}".format(k, accuracies[k], f1_scores[k]))

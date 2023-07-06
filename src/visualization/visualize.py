import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import ast


df = pd.read_csv("data/processed/data_batch_1.csv")

first_image = df.iloc[0]["data"]
first_image = ast.literal_eval(first_image)
print(len(first_image))
image = np.reshape(first_image, (32, 32, 3), order='F')
image = np.transpose(image, (1, 0, 2))

plt.imshow(image)
plt.show()

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import ast
from skimage.feature import hog
from skimage import exposure, color

df = pd.read_csv("data/processed/data_batch_1.csv")

first_image = df.iloc[0]["data"]
first_image = ast.literal_eval(first_image)
print(len(first_image))
image = np.reshape(first_image, (32, 32, 3), order='F')
image = np.transpose(image, (1, 0, 2))
image = color.rgb2gray(image)

descriptor, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.figure(figsize=(8, 8))
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.title('Histogram of Oriented Gradients')
plt.axis('off')
plt.show()
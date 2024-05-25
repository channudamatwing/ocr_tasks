import numpy as np
import pandas as pd
import os
import cv2

train_images_path = "experiments/name_handwriting/train_v2/train"
train_df = pd.read_csv('experiments/name_handwriting/written_name_train_v2.csv')
filenames = train_df['FILENAME']
labels = train_df['IDENTITY']
images = []
for filename in filenames:
    images.append(cv2.imread(f"{train_images_path}/{filename}"))
images = np.asarray(images, dtype="object")
print(images[0])

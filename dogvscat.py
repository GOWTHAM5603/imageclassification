import numpy as np
import os
import cv2 
import random
import matplotlib.pyplot as plt
import pickle

DIRECTORY = r'D:\classification\dogscats\dogscats\dataset'
CATEGORIES=['cat','dog']

IMG_SIZE = 100
data = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])

len(data)

random.shuffle(data)

x = []
y = []

for features, labels in data:
    x.append(features)
    y.append(labels)

x = np.array(x)
y = np.array(y)

pickle.dump(x, open('x.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))


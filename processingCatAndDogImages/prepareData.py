import numpy as np
import cv2
import os
import random
import pickle


DIRECTORY = r'<train_data_path>'
CATEGORIES = ['cats', 'dogs']

IMG_SIZE = 100

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    category_index = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append([img_array, category_index])

random.shuffle(data)

X = []
Y = []

for img, category_index in data:
    X.append(img)
    Y.append(category_index)

X = np.array(X)
Y = np.array(Y)

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(Y, open('Y.pkl', 'wb'))

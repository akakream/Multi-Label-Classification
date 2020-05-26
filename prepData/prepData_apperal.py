import pickle
import os
import numpy as np
import cv2
import random

labels = ['black', 'blue', 'brown', 'green', 'red', 'white', 'dress', 'shirt', 'shorts', 'pants', 'shoes']

training_data = []

X_TRAIN = []
Y_TRAIN = []
IMAGE_SIZE = 256

for folder in os.listdir('../data/'):
    f_name = folder.split('_')
    color_index = f_name[0]
    color_index = labels.index(color_index)
    cloth_index = f_name[1]
    cloth_index = labels.index(cloth_index)
    label = np.zeros((11,), dtype=int)
    label[color_index] = 1
    label[cloth_index] = 1
    for img in os.listdir(f'../data/{folder}'):
        img_array = cv2.imread(f'../data/{folder}/{img}', cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_array, (IMAGE_SIZE,IMAGE_SIZE))
        training_data.append([img_resized, label])


random.shuffle(training_data)

for img, label in training_data:
    X_TRAIN.append(img)
    Y_TRAIN.append(label)

X_TRAIN = np.array(X_TRAIN).reshape(-1, 256, 256, 3)                        
Y_TRAIN = np.array(Y_TRAIN)

pickle_out = open('X_TRAIN.pickle', 'wb')
pickle.dump(X_TRAIN, pickle_out)
pickle_out.close()

pickle_out = open('Y_TRAIN.pickle', 'wb')
pickle.dump(Y_TRAIN, pickle_out)
pickle_out.close()


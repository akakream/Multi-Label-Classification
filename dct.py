import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import model
import pickle
import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Each image measures 256x256 pixels.
IMG_SIZE = 256
# rgb
CHANNELS = 3

'''
There are originally 21 classes in the 'UC Merced Land Use Dataset' and 100 images for each class.

70 training instances for each class
15 test instances for each class
15 validation instances for each class

Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," 
ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.
'''
TRAINING_DIR = './ucMerced/Splits/training'
TEST_DIR = './ucMerced/Splits/test'
VALIDATION_DIR = './ucMerced/Splits/validation'

'''
17 CLASSSES that were used for multi-label annotation by RSIM
B. Chaudhuri, B. Demir, S. Chaudhuri, L. Bruzzone, "Multi-label Remote Sensing Image Retrieval using a 
Semi-Supervised Graph-Theoretic Method", IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no.1, 2018.
'''
CATEGORIES = ['airplane', 'bare-soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass', 'mobile-home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']


def plot():
    print('--plot--')

def main():

    # Loads the dataset
    X_TRAIN = pickle.load(open("pickleRick/X_TRAIN.pickle", "rb"))
    Y_TRAIN = pickle.load(open("pickleRick/Y_TRAIN.pickle", "rb"))
    X_TEST = pickle.load(open("pickleRick/X_TEST.pickle", "rb"))
    Y_TEST = pickle.load(open("pickleRick/Y_TEST.pickle", "rb"))
    X_VALIDATION = pickle.load(open("pickleRick/X_VALIDATION.pickle", "rb"))
    Y_VALIDATION = pickle.load(open("pickleRick/Y_VALIDATION.pickle", "rb"))
    # x_train = tf.keras.utils.normalize(x_train, axis=1)
    # x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    X_TRAIN = X_TRAIN/255.0
    X_TEST = X_TEST/255.0
    X_VALIDATION = X_VALIDATION/255.0
    #print(f'X_TRAIN:{X_TRAIN[0]}')
    #print(f'Y_TRAIN:{Y_TRAIN[0]}')

    dct_model = model.buildModel()
    model.train(dct_model, X_TRAIN, Y_TRAIN)
    model_sum = dct_model.summary()
    print(f'model summary: {model_sum}')
    model.eval(dct_model, X_TEST, Y_TEST)
    try:
        model.predict(dct_model, X_TEST)
    except:
        print("model.predict gave an error")
    model.saveModel(dct_model)  
    
    # preds = dct_model.predict(X_test)
    # preds[preds>=0.5] = 1
    # preds[preds<0.5] = 0

if __name__ == '__main__':
    main()

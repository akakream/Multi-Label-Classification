import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import model
import pickle
import datetime
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def add_arguments():
    ap = argprase.ArgumentParser()
    ap.add_argument('-d', '--data_destination', required=True, help='relative destination to the folder where pickle files are located')
    ap.add_argument('-s', '--shape', required=True, help='Input shape for the model. It is usually (256,256,3), if it is rgb')
    ap.add_argument('-c', '--classes', required=True, help='Number of classes. This is going to be added to the last layer of the model')
    ap.add_argument('-b', '--batch_size', default=64, help='Batch size, default is 64')
    ap.add_argument('-e', '--epochs', default=10, help='Number of epochs, default is 10')
    args = vars(ap.parse_args())

def plot():
    print('--plot--')

def main(data_dest, shape, classes, batchsize, epochs):

    # Loads the dataset
    X_TRAIN = pickle.load(open(f"{data_dest}/X_TRAIN.pickle", "rb"))
    Y_TRAIN = pickle.load(open(f"{data_dest}/Y_TRAIN.pickle", "rb"))
    X_TEST = pickle.load(open(f"{data_dest}/X_TEST.pickle", "rb"))
    Y_TEST = pickle.load(open(f"{data_dest}/Y_TEST.pickle", "rb"))
    X_VALIDATION = pickle.load(open(f"{data_dest}/X_VALIDATION.pickle", "rb"))
    Y_VALIDATION = pickle.load(open(f"{data_dest}/Y_VALIDATION.pickle", "rb"))
    
    #normalize
    X_TRAIN = X_TRAIN/255.0
    X_TEST = X_TEST/255.0
    X_VALIDATION = X_VALIDATION/255.0

    dct_model = model.buildModel(shape, classes)
    model.train(dct_model, X_TRAIN, Y_TRAIN, batchsize, epochs)
    model_sum = dct_model.summary()
    print(f'model summary: {model_sum}')
    model.eval(dct_model, X_TEST, Y_TEST, batchsize)
    try:
        model.predict(dct_model, X_TEST)
    except:
        print("model.predict gave an error")
    model.saveModel(dct_model)  
    
    # preds = dct_model.predict(X_test)
    # preds[preds>=0.5] = 1
    # preds[preds<0.5] = 0

if __name__ == '__main__':
    add_arguments()
    main(args['data_destination'], args['shape'], args['classes'], args['batch_size'], args['epochs'])

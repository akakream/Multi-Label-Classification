from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
import argparse

def add_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data_destination', required=True, help='relative destination to the folder where pickle files are located')
    ap.add_argument('-s', '--shape', required=True, help='Input shape for the model. It is usually (256,256,3), if it is rgb')
    ap.add_argument('-c', '--classes', required=True, help='Number of classes. This is going to be added to the last layer of the model')
    ap.add_argument('-b', '--batch_size', default=64, help='Batch size, default is 64')
    ap.add_argument('-e', '--epochs', default=10, help='Number of epochs, default is 10')
    args = vars(ap.parse_args())

    return args

# data_destination, shape, classes, batch_size, epochs
def main(args):
    
    new_input = Input(shape=tuple(int(dim) for dim in args['shape'].split(',')))

    X_TRAIN = pickle.load(open(f'{args["data_destination"]}/X_TRAIN.pickle', 'rb'))
    Y_TRAIN = pickle.load(open(f'{args["data_destination"]}/Y_TRAIN.pickle', 'rb'))

    X_TRAIN = X_TRAIN / 255.0

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=new_input, classes=int(args['classes']))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(int(args['classes']), activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    '''
    for layer in base_model.layers:
        layer.trainable = False
    '''

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

    model.fit(X_TRAIN, Y_TRAIN, batch_size=int(args['batch_size']), epochs=int(args['epochs']))

    model.summary()

    #PREDICTION

    '''
    img_path = 'ucMerced/Splits/test/airplane/airplane04.tif'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    '''

if __name__ == '__main__':
    args = add_arguments()
    main(args)

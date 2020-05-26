from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input

new_input = Input(shape=(256,256,3))

X_TRAIN = pickle.load(open('../dr_hax/pickleRick/X_TRAIN.pickle', 'rb'))
Y_TRAIN = pickle.load(open('../dr_hax/pickleRick/Y_TRAIN.pickle', 'rb'))

X_TRAIN = X_TRAIN / 255.0

base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=new_input, classes=11)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(11, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

'''
for layer in base_model.layers:
    layer.trainable = False
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

model.fit(X_TRAIN, Y_TRAIN, batch_size=256, epochs=4)

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


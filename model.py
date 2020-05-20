import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AvgPool3D, BatchNormalization
from tensorflow.keras.optimizers import SGD

# TODO: Build a second model to collaborate

# TODO: Choose a selection strategy and apply

# This is called L1 in the paper, aka Classification Loss
def buildClassLoss():
    loss = 0
    print('Classification Loss')
    return loss    

# This is called L2 in the paper, aka Discrepancy Loss 
def buildDiscLoss():
    loss = 0
    print('Discrepancy Loss')
    return loss    

# This is called L3 in the paper, aka Consistency Loss 
def buildConsLoss():
    loss = 0
    print('Consistency Loss')
    return loss

# Loss = L1 + (lambda3)*L3 - (lambda2)*L2  ---> Do this for both F and G, which are the two complementary networks
def buildFinalLoss(L1, L2, L3):
    print('Loss = L1 + (lambda3)*L3 - (lambda2)*L2')

def buildModel():
    model = Sequential()

    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(256,256,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu'))    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # TODO: Add here the L2 layer
    buildDiscLoss()

    model.add(Flatten())
    model.add(Dense(17, activation=tf.nn.sigmoid))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: Add here the L3 layer
    buildConsLoss()

    return model

def train(model, X_TRAIN, Y_TRAIN):
    model.fit(X_TRAIN, Y_TRAIN, batch_size=32, epochs=10)    

def eval(model, X_TEST, Y_TEST):
    val_loss, val_acc = model.evaluate(X_TEST, Y_TEST, batch_size=32)
    print(f'val_loss is {val_loss} and val_acc is {val_acc}')    

def predict(model, X_TEST):
    print (model.predict(X_TEST[1,:]))

def saveModel(model):
    model.save('DCT_UC_Merced')

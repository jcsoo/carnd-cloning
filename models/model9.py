
# Model 9 - Boost Correction Factor to 0.3
# Epoch 3/3
# 18014/18014 [==============================] - 48s - loss: 0.0083 - val_loss: 0.0411

import sys
import loader
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D

def build_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def main(args):
    model = build_model()

    data_path = args[0]
    X_train, y_train = loader.augment_flipped(loader.load_training_sets_all(args, correction=0.3))
    print("%d, %d" % (len(X_train), len(y_train)))

    model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)
    model.save('model9.h5')


if __name__=='__main__':
    main(sys.argv[1:])

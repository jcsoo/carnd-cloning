import sys
import loader
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator

def normalize(x):
    import tensorflow as tf
    return tf.nn.l2_normalize(x, 2)

def build_model():
    model = Sequential()
    #model.add(Cropping2D(cropping=((25,10), (0,0)), input_shape=(80, 160, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66, 200, 3)))
    #model.add(Lambda(normalize))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.1))    
    model.add(Flatten())
    model.add(Dense(100))

    model.add(Dense(50))
    #model.add(Dropout(0.1))
    model.add(Dense(10))
    #model.add(Dropout(0.1))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

SCALE_Y = None
EXCLUDE_Y = 0.01
SIZE = (200, 66)
CSPACE = 'yuv'

def main(args):
    model = build_model()
    
    samples = loader.load_data(args, exclude_y=EXCLUDE_Y)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('train_samples:', len(train_samples), 'validation_samples:', len(validation_samples))
    
    tg = loader.training_generator(train_samples, batch_size=32, scale_y=SCALE_Y, cspace=CSPACE, size=SIZE)
    vg = loader.validation_generator(validation_samples, batch_size=32, scale_y=SCALE_Y, cspace=CSPACE, size=SIZE)
    
    model.fit_generator(tg,
                        samples_per_epoch=len(train_samples) * 2,
                        validation_data=vg,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=8)
    model.save('model27.h5')

if __name__=='__main__':
    main(sys.argv[1:])

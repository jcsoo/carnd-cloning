import sys
import loader
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, ELU
from keras.preprocessing.image import ImageDataGenerator

# Level 2 Normalize channel 2 (V for YUV / HSV)
# This should reduce the differences in brightness when in shadow.
def normalize(x):
    import tensorflow as tf
    return tf.nn.l2_normalize(x, 2)

# Build the Keras model
def build_model():
    model = Sequential()
    # Remove the top 80 and bottom 30 pixels, leaving a vertical height of
    # 50 pixels. This should leave the part of the road immediately in front
    # of the vehicle.
    model.add(Cropping2D(cropping=((80, 30), (0,0)), input_shape=(160, 320, 3)))
    # Convert from [0..255] to [-1.0..1.0]
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    # Apply Level 2 normalization to the V channel
    model.add(Lambda(normalize))
    # Three 5x5 convolutions. The third layer does not subsample because the
    # vertical size of the image is too small.
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    # Two 3x3 convolutions
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # One dropout layer to reduce overfitting
    model.add(Dropout(0.1))
    model.add(Flatten())
    # Three fully connected layers with RELU activation
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Output layer
    model.add(Dense(1))

    # Use Mean Squared Error loss function and Adam optimizer,
    # learning rate reduced to 0.0001
    model.compile(loss='mse', optimizer='adam', lr=0.0001)
    return model

# Parameters for sample loading and processing

SCALE_Y = None       # No scaling steering angle
EXCLUDE_Y = 0.005    # Exclude samples with abs(steering) < 0.005
SIZE = (320, 160)    # Image size 320x160 (not scaled)
CSPACE = 'yuv'       # Convert to YUV colorspace
LEFT = False         # Do not include left camera image
RIGHT = False        # Do not include right camera image
EPOCHS = 5           # Number of epochs for training
def main(args):
    model = build_model()
    
    samples = loader.load_data(args, exclude_y=EXCLUDE_Y)

    # Use an 80 / 20 split for training / validation samples
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('train_samples:', len(train_samples), 'validation_samples:', len(validation_samples))
    
    tg = loader.training_generator(train_samples, batch_size=32, correction=0.1, left=LEFT, right=RIGHT, scale_y=SCALE_Y, cspace=CSPACE, size=SIZE)
    vg = loader.validation_generator(validation_samples, batch_size=32, scale_y=SCALE_Y, cspace=CSPACE, size=SIZE)
    
    model.fit_generator(tg,
                        samples_per_epoch=len(train_samples) * 2,
                        validation_data=vg,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=EPOCHS)

    # Save model
    model.save('model.h5')

if __name__=='__main__':
    main(sys.argv[1:])

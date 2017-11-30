import sys
import loader
from sklearn.model_selection import train_test_split
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


SIZE = (160, 80)

def main(args):
    model = build_model()
    
    samples = loader.load_data(args, exclude_y=0.1)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('train_samples:', len(train_samples), 'validation_samples:', len(validation_samples))
    
    tg = loader.training_generator(train_samples, batch_size=32, correction=0.25, scale_y=1.1, size=SIZE)
    vg = loader.validation_generator(validation_samples, batch_size=32, scale_y=1.1, size=SIZE)
    
    model.fit_generator(tg,
                        samples_per_epoch=len(train_samples) * 2,
                        validation_data=vg,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=3)
    model.save('model15.h5')

if __name__=='__main__':
    main(sys.argv[1:])

import sys
import loader
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def main(args):
    model = build_model()

    data_path = args[0]
    X_train, y_train = loader.load_training_set(data_path)
    print("%d, %d" % (len(X_train), len(y_train)))

    model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True)
    model.save('model2.h5')


if __name__=='__main__':
    main(sys.argv[1:])

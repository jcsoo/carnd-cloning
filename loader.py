import sys, os, csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

# Reads and parses CSV file, adjusting image paths and optionally excluding
# records with a abs(steering) < exclude_y
def load_data(paths, exclude_y=None):
    records = []    
    for path in paths:
        print('load_data:', path)
        base_dir = os.path.split(path)[0]
        with open(path) as f:
            reader = csv.reader(f)
            for line in reader:            
                record = {
                    'img_center': os.path.join(base_dir, 'IMG', os.path.split(line[0])[1]),
                    'img_left': os.path.join(base_dir, 'IMG', os.path.split(line[1])[1]),
                    'img_right': os.path.join(base_dir, 'IMG', os.path.split(line[2])[1]),
                    'steering' : float(line[3]),
                    'throttle' : float(line[4]),
                    'brake' : float(line[5]),
                    'speed' : float(line[6]),
                }
                if exclude_y and abs(record['steering']) < exclude_y:
                    continue
                records.append(record)
    return records

# Loads an image from a path, optionally converting colorspace and resizing.
def load_image_rgb(path, size=None, cspace=None):
    if cspace == 'hsv':
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
    elif cspace == 'yuv':
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YUV)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

# Load a single training sets, returning an NP array of
# images and steering angles.
def load_training_set(path):
    return load_training_sets([path])

# Load a list of training sets, returning an NP array of
# images and steering angles.
def load_training_sets(paths):    
    images = []
    measurements = []
    for record in load_data(paths):
        images.append(load_image_rgb(record['img_center']))
        measurements.append(record['steering'])
    return np.array(images), np.array(measurements)

# Load a training set including left + right camera angles.
# Optionally, resize images and add / subtract a steering angle correction
# for the left + right steering images.
def load_training_set_all(path, correction=0.2, size=None):
    return load_training_sets_all([path], correction=correction, size=size)

def load_training_sets_all(paths, correction=0.2, size=None):    
    images = []
    measurements = []
    records = load_data(paths)
    print('Records:', len(records))
    for record in records:
        steering = record['steering']
        
        images.append(load_image_rgb(record['img_center'], size=size))
        measurements.append(steering)

        # Images from left camera have rightward correction
        images.append(load_image_rgb(record['img_left'], size=size))
        measurements.append(steering + correction)

        # Images from right camera have leftward correction
        images.append(load_image_rgb(record['img_right'], size=size))
        measurements.append(steering - correction)
    return np.array(images), np.array(measurements)

# Returns a generator that yields batches of images and target values. Optionally performs the following
# transformations:
#
#  scale_y - multiply all steering angles by a constant factor
#  correction, left, right - include left and right camera images with associated correction factor
#  size - resize the output images to `size`
#  cspace - change the output color space to 'yuv' or 'hsv'
#
#  NOTE: All output images and targets are automatically augmented by their horizontally flipped values.
def training_generator(samples, batch_size=32, correction=0.2, scale_y=None, size=None, left=False, right=False, cspace=None):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for record in batch_samples:
                steering = record['steering']
                if scale_y:
                    steering *= scale_y

                images.append(load_image_rgb(record['img_center'], size=size, cspace=cspace))
                measurements.append(steering)

                if left:
                    # Images from left camera have rightward correction
                    images.append(load_image_rgb(record['img_left'], size=size, cspace=cspace))
                    measurements.append(steering + correction)

                if right:
                    # Images from right camera have leftward correction
                    images.append(load_image_rgb(record['img_right'], size=size, cspace=cspace))
                    measurements.append(steering - correction)            
            
            yield augment_flipped((np.array(images), np.array(measurements)))

# Experiment with Keras ImageDataGenerator
def training_generator2(samples, batch_size=32, correction=0.2, scale_y=None, size=None, left=False, right=False, cspace=None):
    images = []
    measurements = []
    for record in samples:
        steering = record['steering']
        if scale_y:
            steering *= scale_y            
        images.append(load_image_rgb(record['img_center'], size=size, cspace=cspace))
        measurements.append(steering)
    images, measurements = np.array(images), np.array(measurements)
    idg = ImageDataGenerator()
    return idg.flow(images, measurements)

    #for (x,y) in idg.flow(images, measurements):
    #    yield x, y
            
# Returns a generator that yields batches of images and target values. Optionally performs the following
# transformations: 
#
#  scale_y - multiply all steering angles by a constant factor
#  size - resize the output images to `size`
#  cspace - change the output color space to 'yuv' or 'hsv'
#
# The transformations used by this generator should match those used by the training generator.
def validation_generator(samples, batch_size=32, size=None, scale_y=None, cspace=None):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for record in batch_samples:
                steering = record['steering']
                if scale_y:
                    steering *= scale_y

                images.append(load_image_rgb(record['img_center'], size=size, cspace=cspace))
                measurements.append(steering)
                
            yield np.array(images), np.array(measurements)
            

# Flip an image and target horizontally.
def augment_flipped(ts):
    (X_train, y_train) = ts
    (X_flipped, y_flipped) = (np.fliplr(X_train), -y_train)
    return np.concatenate((X_train, X_flipped)), np.concatenate((y_train, y_flipped))

# Test load_training_set
def main(args):
    data_path = args[0]
    X_train, y_train = load_training_set(data_path)
    print("%d, %d" % (len(X_train), len(y_train)))


if __name__=='__main__':
    main(sys.argv[1:])

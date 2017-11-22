import sys, os, csv
import cv2
import numpy as np

def load_data(path):
    path = os.path.join(path)
    base_dir = os.path.split(path)[0]
    records = []
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
            records.append(record)
    return records

def load_image_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def load_training_set(path):
    return load_training_sets([path])

def load_training_sets(paths):    
    images = []
    measurements = []
    for path in paths:
        for record in load_data(path):
            images.append(load_image_rgb(record['img_center']))
            measurements.append(record['steering'])
    return np.array(images), np.array(measurements)

def load_training_set_all(path, correction=0.2):
    return load_training_sets_all([path], correction=correction)

def load_training_sets_all(paths, correction=0.2):    
    images = []
    measurements = []
    for path in paths:
        for record in load_data(path):
            steering = record['steering']
        
            images.append(load_image_rgb(record['img_center']))
            measurements.append(steering)

            # Images from left camera have rightward correction
            images.append(load_image_rgb(record['img_left']))
            measurements.append(steering + correction)

            # Images from right camera have leftward correction
            images.append(load_image_rgb(record['img_right']))
            measurements.append(steering - correction)
            
    return np.array(images), np.array(measurements)


def augment_flipped(ts):
    (X_train, y_train) = ts
    (X_flipped, y_flipped) = (np.fliplr(X_train), -y_train)
    return np.concatenate((X_train, X_flipped)), np.concatenate((y_train, y_flipped))

def main(args):
    data_path = args[0]
    X_train, y_train = load_training_set(data_path)
    print("%d, %d" % (len(X_train), len(y_train)))


if __name__=='__main__':
    main(sys.argv[1:])

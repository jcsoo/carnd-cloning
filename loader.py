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
    images = []
    measurements = []
    for record in load_data(path):
        images.append(load_image_rgb(record['img_center']))
        measurements.append(record['steering'])

    return np.array(images), np.array(measurements)

def main(args):
    data_path = args[0]
    X_train, y_train = load_training_set(data_path)
    print("%d, %d" % (len(X_train), len(y_train)))


if __name__=='__main__':
    main(sys.argv[1:])

import os
import numpy as np

from scipy.misc import imread

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from random import randint

from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, GlobalMaxPooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras import backend as K

from keras.callbacks import LambdaCallback, ProgbarLogger

from keras.optimizers import Adam

from run_tests import read_csv, save_csv

from random import randint

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout

from keras.models import Model, Sequential
from keras.optimizers import Nadam
from keras.layers import BatchNormalization, Convolution2D, Input, merge, LeakyReLU, MaxPooling2D
from keras.layers.core import Activation, Layer
from keras.optimizers import SGD, Adam

def SimpleCNN(withDropout=False):
    '''
    WithDropout: If True, then dropout regularlization is added.
    This feature is experimented later.
    '''
    model = Sequential()
    model.add(Conv2D(32,(3, 3), input_shape = (100, 100, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    if withDropout:
        model.add(Dropout(0.1))
        
    model.add(Conv2D(64,(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    if withDropout:
        model.add(Dropout(0.1))
    
    model.add(Conv2D(128,(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if withDropout:
        model.add(Dropout(0.1))
        
    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(Activation('relu'))
    if withDropout:
        model.add(Dropout(0.1))
        
    model.add(Dense(256))
    model.add(Activation('relu'))
    if withDropout:
        model.add(Dropout(0.1))
        
    model.add(Dense(28))
    sgd = Adam(decay = 1e-6)
    model.compile(loss="mean_squared_error",optimizer=sgd)
    return(model)

def generator(train_img_dir, dataset, batch_size = 1):
    x_batch = np.zeros((batch_size, 100, 100, 3))
    y_batch = []
    i = 0
    full_indx = np.array([i for i in range(len(dataset))])
    begin_batch = 0
    while True:
        filename_idxs = []
        
        if begin_batch + batch_size < len(dataset):
            filename_idxs = full_indx[begin_batch : begin_batch + batch_size]
            begin_batch += batch_size
        else:
            filename_idxs = full_indx[begin_batch : len(dataset)]
            begin_batch = (begin_batch + batch_size) % len(dataset)
            filename_idxs += full_indx[0 : begin_batch]

        filenames = [list(dataset.keys())[filename_idx] for filename_idx in filename_idxs]
        for i in range(batch_size):
            image = load_img(train_img_dir + '/' +  filenames[i], target_size = (100, 100))
            image -= np.mean(image, keepdims=True)
            image /= (np.std(image, keepdims=True) + 0.00001)
            rows, cols, channels = imread(test_img_dir + '/' +  filename).shape
            x_batch[i] = image
            y = dataset[filenames[i]]
            for j in range(14):
                y[j * 2] *= 100 / col
                y[j * 2 + 1] *= 100 / rows
            y_batch.append(y)
        yield x_batch, y_batch


def train_detector(train_gt, train_img_dir, fast_train=True):
    '''
    train_gt - csv file with labels
    train_img_dir - path to dir with imgs
    fast_train - fast train for testing at cv-gml
    '''
    model = SimpleCNN()
    
    if fast_train:
        #dataset = train_gt
        #model.fit_generator(generator(train_img_dir = train_img_dir, dataset = dataset), steps_per_epoch = 1, epochs = 1, verbose = 0) 
        return model
    
    
    # main model training, prep_dirs should be made just one time
    
    batch_sizes = [32, 64, 128, 256, 512]
    
    for batch_size in batch_sizes:
        model.fit_generator(generator(train_img_dir = train_img_dir, dataset = dataset), steps_per_epoch = 5000, epochs = 3, verbose = 0)
    
    return model


def detect(model, test_img_dir):
    res = {}
    for filename in os.listdir(test_img_dir):
        if filename.endswith(".jpg"):
            image = load_img(test_img_dir + '/' +  filename, target_size = (100, 100))
            image -= np.mean(image, keepdims=True)
            image /= (np.std(image, keepdims=True) + 0.00001)
            rows, cols = imread(test_img_dir + '/' +  filename).shape[:2]
            ans = model.predict(image.reshape((1, 100, 100, 3))).flatten()
            for j in range(14):
                ans[j * 2] *= cols / 100
                ans[j * 2 + 1] *= rows / 100
            res[filename] = list(ans)
        else:
            continue
    return res
        
    



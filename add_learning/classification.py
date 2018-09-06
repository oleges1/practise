import os
import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img
from random import randint

from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, GlobalMaxPooling2D, Dropout
from keras import backend as K

from keras.callbacks import LambdaCallback, ProgbarLogger

from keras.optimizers import Adam

from run_tests import read_csv, save_csv

from skimage.transform import resize
from random import randint

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res

def rotare(img, times):
    if times == 0:
      return img
    if times == 1:
      return np.rot90(img)
    if times == 2:
      return np.rot90(np.rot90(img))
    if times == 3:
      return np.rot90(np.rot90(np.rot90(img)))


def generator(train_img_dir, dataset, batch_size = 1, augmentations = False):
    x_batch = np.zeros((batch_size, 299, 299, 3))
    y_batch = np.zeros((batch_size, 50))
    i = 0
    full_indx = np.array([i for i in range(len(dataset))])
    begin_batch = 0
    while True:
        #filename_idxs = np.random.choice(len(dataset), batch_size)
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
            image = load_img(train_img_dir + '/' +  filenames[i], target_size = (299, 299))
            if augmentations:
                seed = randint(0, 3)
                image = rotare(image, seed)
            image -= np.mean(image, keepdims=True)
            image /= (np.std(image, keepdims=True) + 0.00001)
            x_batch[i] = image
            y_batch[i][dataset[filenames[i]]] = 1
        yield x_batch, y_batch
        

def make_generator(path, batch_size = 32, batches_per_training = 10, target_size = (299, 299)):
    data_gen_args = dict(samplewise_center = True,
                         samplewise_std_normalization = True,
                         horizontal_flip = True,
                         rotation_range=90.)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    train_generator = image_datagen.flow_from_directory(path, target_size = target_size, batch_size = batches_per_training * batch_size)
    
    return train_generator


def train_classifier(train_gt, train_img_dir, fast_train=True, prep_dirs = False):
    '''
    train_gt - csv file with labels
    train_img_dir - path to dir with imgs
    fast_train - fast train for testing at cv-gml
    prep_dirs - flag for preparing dirs for flow_from_directory
    '''
    if prep_dirs:
        import shutil
        
        dataset = train_gt
        file_names = dataset.keys()
        img_labels = dataset.values()

        folders_to_be_created = np.unique(img_labels)

        source = os.getcwd()

        for new_path in folders_to_be_created:
            if not os.path.exists(train_img_dir + '/' +  str(new_path)):
                os.makedirs(train_img_dir + '/' +  str(new_path))

        folders = folders_to_be_created.copy()

        for f in range(len(file_names)):
            current_img = file_names[f]
            current_label = img_labels[f]
            shutil.move(train_img_dir + '/' + str(current_img), train_img_dir + '/' + str(current_label))
    
    base_model = InceptionV3(weights='imagenet', include_top = False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalMaxPooling2D()(x)

    # let's add a fully-connected layer with dropout
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(50, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers, I don't why but freezing this layers stops learning on validation
    #for layer in base_model.layers:
    #    layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable), also set small learning_rate
    model.compile(optimizer=Adam(lr=0.00004), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if fast_train:
        dataset = train_gt
        model.fit_generator(generator(train_img_dir = train_img_dir, dataset = dataset), steps_per_epoch = 1, epochs = 1, verbose = 0) 
        return model
    
    
    # main model training, prep_dirs should be made just one time
    
    batch_sizes = [32, 36, 40, 48, 56, 64, 69, 71]
    
    for batch_size in batch_sizes:
        print('train on batch size:', batch_size)
        train_generator = make_generator(batch_size = batch_size)
        batches = 0
        for x_batch, y_batch in train_generator:
            model.fit(x_batch, y_batch,
              batch_size=batch_size, epochs=1, shuffle=False)
            batches += batches_per_training

            if (batches * batch_size) >= 2500:
                break
    
    #model.save('birds_model.hdf5')
    return model


def classify(model, test_img_dir, verbose = 0, true_labels = None):
    #model = load_model('birds_model.hdf5')
    res = {}
    for filename in os.listdir(test_img_dir):
        if filename.endswith(".jpg"):
            image = load_img(test_img_dir + '/' +  filename, target_size = (299, 299))
            image -= np.mean(image, keepdims=True)
            image /= (np.std(image, keepdims=True) + 0.00001)
            ans = int((model.predict(image.reshape((1, 299, 299, 3)))).argmax())
            if verbose:
                if true_labels:
                    print(filename, 'predict:', ans, 'true:', true_labels[filename])
                else:
                    print(filename, 'predict:', ans)
            res[filename] = ans
        else:
            continue
    return res
        
    


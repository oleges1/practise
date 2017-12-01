#!/usr/bin/python3

from sys import argv, exit
from numpy import zeros, save, load
from os.path import join
from fit_and_classify import fit_and_classify, extract_hog
from skimage.io import imread

def read_gt(gt_dir):
    fgt = open(join(gt_dir, 'gt.csv'))
    next(fgt)
    lines = fgt.readlines()

    filenames = []
    labels = zeros(len(lines))
    for i, line in enumerate(lines):
        filename, label = line.rstrip('\n').split(',')
        filenames.append(filename)
        labels[i] = int(label)

    return filenames, labels


def extract_features(path, filenames):
    hog_length = len(extract_hog(imread(join(path, filenames[0]), plugin='matplotlib')))
    data = zeros((len(filenames), hog_length))
    for i in range(0, len(filenames)):
        filename = join(path, filenames[i])
        data[i, :] = extract_hog(imread(filename, plugin='matplotlib'))
    return data

if len(argv) != 3:
    print('Usage: %s train_data_path test_data_path' % argv[0])
    exit(0)

train_data_path = argv[1]
test_data_path = argv[2]

train_filenames, train_labels = read_gt(train_data_path)
test_filenames, test_labels = read_gt(test_data_path)

# train_features = extract_features(train_data_path, train_filenames)
# save("train_features.npy", train_features)
train_features = load("train_features.npy")
# test_features = extract_features(test_data_path, test_filenames)
# save("test_features.npy", test_features)
test_features = load("test_features.npy")

y = fit_and_classify(train_features, train_labels, test_features)
print('Accuracy: %.4f' % (sum(test_labels == y) / float(test_labels.shape[0])))

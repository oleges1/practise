import numpy as np

from sys import argv, stderr, stdout, exit
from os import mkdir
from os.path import dirname, exists
from glob import iglob
from skimage.io import imread, imsave
from pickle import load
from time import clock
import csv

from seam_carve import seam_carve

def seam_coords(seam_mask):
    coords = np.where(seam_mask)
    t = [i for i in zip(coords[0], coords[1])]
    t.sort(key = lambda i: i[0])
    return tuple(t)

def msk_to_std(msk):
    return ((msk[:,:,0]!=0)*(-1) + (msk[:,:,1]!=0)).astype('int8')

if len(argv) != 3:
    stderr.write('Usage: %s mode input_dir\n' % argv[0])
    exit(1)

if argv[1] != '--base' and argv[1] != '--full':
    stderr.write('Usage: %s mode input_dir\n' % argv[0])
    exit(1)

mode = argv[1][2:]
input_dir = argv[2]
output_dir = 'res'

number = 0
TP = 0
for filename in iglob(input_dir + '/*.png'):
    time = clock()
    if filename.find('_mask') >= 0:
        continue    

    print(filename)

    img = imread(filename)
    if mode == 'full':
        msk = imread(filename[:-4] + '_mask.png')
        msk = msk_to_std(msk)
        number += 8
    else:
        msk = None
        number += 2

    file = open(filename[:-4] + '_seams', 'rb')
    name = filename.split('/')[1]
    name = name.split('.')[0]

    if mode == 'base':
        for orientation in ('horizontal', 'vertical'):
            my_answer = seam_coords(seam_carve(img, orientation + ' shrink')[2])
            res = np.zeros(img.shape)
            for pair in my_answer:
                res[pair[0], pair[1], 0] = 1
            correct_answer = load(file)
            for pair in correct_answer:
                res[pair[0], pair[1], 1] = 1
            out_filename = output_dir + '/' + name + '_shrink_' + orientation + '.png'
            out_dirname = dirname(out_filename)
            if not exists(out_dirname):
                mkdir(out_dirname)
            imsave(out_filename, res)
            TP += correct_answer == my_answer

    elif mode == 'full':
        for m in (None, msk):
            for direction in ('shrink', 'expand'):
                gopa = 0
                for orientation in ('horizontal', 'vertical'):
                    my_answer = seam_coords(seam_carve(img, orientation + ' ' + direction, mask = m)[2])
                    res = img.copy()
                    for pair in my_answer:
                        res[pair[0], pair[1], 0] = 1
                    correct_answer = load(file)
                    for pair in correct_answer:
                        if (res[pair[0], pair[1], 0] != 1):
                            gopa = 1
                        res[pair[0], pair[1], 1] = 1
                    if (m is None):
                        out_filename = output_dir + '/' + name + "_" + direction + "_" + orientation + '.png'
                    else:
                        out_filename = output_dir + '/' + name + "_" + direction + "_" + orientation + '+mask.png'
                    out_dirname = dirname(out_filename)
                    if not exists(out_dirname):
                        mkdir(out_dirname)
                    imsave(out_filename, res)
                    if (gopa == 1):
                        print('proebalsya')
                    TP += correct_answer == my_answer

    file.close()
    print(clock() - time)

print('Accuracy: {0:.2%}'.format(TP / number))


# -*- coding:utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

IMAGE_SIZE = (100, 100)
MEAN_VALUE = 128

image_dir='dataset/images'
# 生成train hdf5
train_file='dataset/train.txt'
train_h5='dataset/HDF5/train.h5'
# 生成val hdf5

val_file='dataset/val.txt'
val_h5='dataset/HDF5/val.h5'

def gen_hdf5(filename,image_dir,h5_path):
    setname, ext = h5_path.split('.')
    with open(filename, 'r') as f:
        lines = f.readlines()

    # np.random.shuffle(lines)

    sample_size = len(lines)
    imgs = np.zeros((sample_size, 1,) + IMAGE_SIZE, dtype=np.float32)
    freqs = np.zeros((sample_size, 2), dtype=np.float32)

    h5_filename = '{}.h5'.format(setname)
    with h5py.File(h5_filename, 'w') as h:
        for i, line in enumerate(lines):
            image_name, fx, fy = line[:-1].split()
            img = plt.imread(os.path.join(image_dir,image_name))[:, :, 0].astype(np.float32)
            img = img.reshape((1, )+img.shape)
            img -= MEAN_VALUE
            imgs[i] = img
            freqs[i] = [float(fx), float(fy)]
            if (i+1) % 1000 == 0:
                print('Processed {} images!'.format(i+1))
        h.create_dataset('data', data=imgs)
        h.create_dataset('freq', data=freqs)

    with open('{}_h5.txt'.format(setname), 'w') as f:
        f.write(h5_filename)


if __name__=='__main__':
    gen_hdf5(train_file,image_dir,train_h5)
    gen_hdf5(val_file,image_dir,val_h5)
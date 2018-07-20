# -*- coding:utf-8 -*-

import sys
import numpy as np
sys.path.append('/path/to/caffe/python')
import caffe
import glob
import os

WEIGHTS_FILE = 'models/letnet5-regression/freq_regression_iter_10000.caffemodel'
DEPLOY_FILE = 'config/letnet5-regression/deploy.prototxt'
test_image='test_image'
mean_file='dataset/mean/train_image_mean.txt'


MEAN_VALUE = []
with open(mean_file) as f:
    ls=f.readlines()
    for l in ls:
        num=float(l)
        MEAN_VALUE.append(num)
#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.array([MEAN_VALUE]))
# transformer.set_raw_scale('data', 255)
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension将图像通道移动到最外层
transformer.set_mean('data', np.array(MEAN_VALUE))            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# image_list = sys.argv[1]
image_list=glob.glob(os.path.join(test_image,'*.jpg'))
batch_size = net.blobs['data'].data.shape[0]
print("batch_size=",batch_size)

i=0
filenames = []
for filename in image_list:
    filenames.append(filename)
    image = caffe.io.load_image(filename)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[0, ...] = transformed_image

    output = net.forward()
    freqs = output['pred']
    pre=freqs[0]
    print('Predicted frequencies for %s is %3.4f and %3.4f '%(filename, pre[0], pre[1]))





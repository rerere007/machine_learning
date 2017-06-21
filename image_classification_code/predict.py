# -*- coding: utf-8 -*-
"""
Created on 2017 6/17
"""

import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from collections import OrderedDict
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time

import numpy as np
from PIL import Image


import six
#import six.moves.cPickle as pickle
from six.moves import queue

import chainer
#import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import *
from chainer import serializers
from package import nin


parser = argparse.ArgumentParser(
    description='Image inspection using chainer')
parser.add_argument('--model','-m',default='modelhdf5', help='Path to model file')
parser.add_argument('--mean', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
args = parser.parse_args()

PATH_TO_MY_DATA = os.path.join('data','processed')
PATH_TO_PROCESSED_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA,'processed_test_images')
PATH_TO_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'test')
PATH_TO_SUBMIT_FILE = 'submit.csv'


def read_image(path, center=False, flip=False):
  image = np.asarray(Image.open(path)).transpose(2, 0, 1)
  if center:
    top = left = cropwidth / 2
  else:
    top = random.randint(0, cropwidth - 1)
    left = random.randint(0, cropwidth - 1)
  bottom = model.insize + top
  right = model.insize + left
  image = image[:, top:bottom, left:right].astype(np.float32)
  image -= mean_image[:, top:bottom, left:right]
  image /= 255
  if flip and random.randint(0, 1) == 0:
    return image[:, :, ::-1]
  else:
    return image


mean_image = np.load(os.path.join("./models/", args.mean))

model = nin.NIN()
serializers.load_hdf5(os.path.join('models', args.model), model)
cropwidth = 256 - model.insize
model.to_cpu()



def predict(net, x):
    h = F.max_pooling_2d(F.relu(net.mlpconv1(x)), 3, stride=2)
    h = F.max_pooling_2d(F.relu(net.mlpconv2(h)), 3, stride=2)
    h = F.max_pooling_2d(F.relu(net.mlpconv3(h)), 3, stride=2)
    h = net.mlpconv4(F.dropout(h, train=net.train))
    h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
    return F.softmax(h)


def create_submit(x, file_name):
    print('predicting ...')
    dic = OrderedDict()
    dic['file_name'] = file_name
    dic['prediction'] = x
    print('done.')
    return pd.DataFrame(dic)

if __name__ == '__main__':
    X = []
    files = os.listdir(PATH_TO_PROCESSED_TEST_IMAGES)
    for f in files: 
        print(os.path.join(PATH_TO_PROCESSED_TEST_IMAGES, f))
        img = read_image(os.path.join(PATH_TO_PROCESSED_TEST_IMAGES, f))
        x = np.ndarray(
            (1, 3, model.insize, model.insize), dtype=np.float32)
        x[0]=img
        x = chainer.Variable(np.asarray(x), volatile='on')

        score = predict(model, x)
        index = np.argmax(score.data[0].tolist())
        X.append(index)
        #categories = np.loadtxt("labels.txt", str, delimiter='\t')
        #categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        #top_k = 5
        #prediction = list(zip(score.data[0].tolist(), categories))
        #prediction.sort(key = lambda x: x, reverse=True)
        #for rank, (score, color) in enumerate(prediction[:top_k], start=1):
        #    print('#%d | %d color | %4.1f%%' % (rank, color, score * 100))
        print("1 rank index is: ", index)
    submit = create_submit(X, files)
    submit.to_csv(PATH_TO_SUBMIT_FILE, index=None, header=None)

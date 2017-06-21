# -*- coding: utf-8 -*-
"""
Created on 2017 6/17
"""

import os
import zipfile
import cv2
from PIL import Image
from package.preprocess_methods import my_preprocess_method
import argparse
import sys

import numpy as np
import chainer

PATH_TO_GIVEN_DATA = os.path.join('data','given')
PATH_TO_MY_DATA = os.path.join('data','processed')
if not os.path.exists(PATH_TO_MY_DATA):
    os.mkdir(PATH_TO_MY_DATA)

PATH_TO_TRAIN_IMAGES = os.path.join(PATH_TO_MY_DATA, 'train')
PATH_TO_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'test')

PATH_TO_DST_TRAIN_IMAGES = os.path.join(PATH_TO_MY_DATA, 'processed_train_images')
PATH_TO_DST_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'processed_test_images')

if not os.path.exists(PATH_TO_DST_TRAIN_IMAGES):
    os.mkdir(PATH_TO_DST_TRAIN_IMAGES)
if not os.path.exists(PATH_TO_DST_TEST_IMAGES):
    os.mkdir(PATH_TO_DST_TEST_IMAGES)

def extract_zipfile(src, dst):
    print('extracting '+src+' to '+dst+' ...')
    with zipfile.ZipFile(src, 'r') as zip_file:
        zip_file.extractall(dst)
    print('done.')

def preprocess_image(path_to_images, path_to_dst_images):
    target_shape = (256, 256)
    output_side_length = 256
    files = os.listdir(path_to_images)
    for f in files:
        print(path_to_images + '/' + f)
        img = cv2.imread(path_to_images + '/' + f)
        height, width, depth = img.shape
        new_height = output_side_length
        new_width = output_side_length

        if height > width:
            new_height = int(output_side_length * height / width)
        else:
            new_width = int(output_side_length * width / height)

        resized_img = cv2.resize(img, (new_width, new_height))
        height_offset = int((new_height - output_side_length) / 2)
        width_offset = int((new_width - output_side_length) / 2)
        cropped_img = resized_img[height_offset:height_offset + output_side_length,width_offset:width_offset + output_side_length]
        cv2.imwrite(path_to_dst_images + '/' + f, cropped_img)


def compute_mean(dataset):
    print('compute mean image')
    sum_image = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    return sum_image / N



if __name__ == '__main__':
    ## extract the given zipfiles(or maybe tarfiles)

    if not os.path.exists(PATH_TO_TRAIN_IMAGES):
        extract_zipfile(os.path.join(PATH_TO_GIVEN_DATA, 'train_images.zip'), PATH_TO_MY_DATA)
        preprocess_image(PATH_TO_TRAIN_IMAGES, PATH_TO_DST_TRAIN_IMAGES)
    if not os.path.exists(PATH_TO_TEST_IMAGES):
        extract_zipfile(os.path.join(PATH_TO_GIVEN_DATA, 'test_images.zip'), PATH_TO_MY_DATA)
        preprocess_image(PATH_TO_TEST_IMAGES, PATH_TO_DST_TEST_IMAGES)


    current_path = os.path.dirname(os.path.abspath(__file__))
    fp = open(PATH_TO_MY_DATA + '/train_master.txt', 'w')

    with open(PATH_TO_GIVEN_DATA + '/train_master.tsv','r') as f:
        for row in f:
            train = row[:-1].split('\t')
            if train[1] == 'category_id': 
                print('skip header')
                continue
            fp.write(current_path + "/" + PATH_TO_DST_TRAIN_IMAGES + "/" + train[0] + " " + train[1] + "\n")

    fp.close() 

    dataset = chainer.datasets.LabeledImageDataset(PATH_TO_MY_DATA + "/train_master.txt", ".")
    mean = compute_mean(dataset)
    np.save("./models" + "/mean.npy", mean)


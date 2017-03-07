#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:58:27 2017

@author: mikkel
"""

import numpy as np
import cv2
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import os
import glob
from Features import *
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read in cars and notcars
srcFolder = '../data/CarND-Vehicle-Detection'
cars = glob.glob(os.path.join(srcFolder,'vehicles','**/*.png'),recursive=True)
notcars = glob.glob(os.path.join(srcFolder,'non-vehicles','**/*.png'),recursive=True)

# Concatenate the two classes
X = cars+notcars

# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

## Extract HOG features

# Tweaked HOG parameters
colorspace = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be [0], [1], [2], a combination, or "ALL"

t=time.time()
feat_train = extract_features_from_image_list(X_train, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,feature_vec=True)
feat_test = extract_features_from_image_list(X_test, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,feature_vec=True)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Fit the normalization function on training data
feat_scaler = StandardScaler().fit(feat_train)

# Apply the normalization to both training and test data
feat_train_scaled = feat_scaler.transform(feat_train)
feat_test_scaled = feat_scaler.transform(feat_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(feat_train_scaled[0]))

# Use a linear SVC ('rbf' kernel is too slow)
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(feat_train_scaled, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(feat_test_scaled, y_test), 4))

# Save the model and its parameters
svc_pickle = {}
svc_pickle['colorspace'] = colorspace
svc_pickle['orient'] = orient
svc_pickle['pix_per_cell'] = pix_per_cell
svc_pickle['cell_per_block'] = cell_per_block
svc_pickle['hog_channel'] = hog_channel
svc_pickle['feat_scaler'] = feat_scaler
svc_pickle['svc'] = svc

pickle.dump( svc_pickle, open( "svc_pickle.p", "wb" ) )
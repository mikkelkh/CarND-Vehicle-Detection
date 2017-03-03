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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read in cars and notcars
srcFolder = '../data/CarND-Vehicle-Detection'
carsImages = glob.glob(os.path.join(srcFolder,'vehicles','**/*.png'),recursive=True)
notcarsImages = glob.glob(os.path.join(srcFolder,'non-vehicles','**/*.png'),recursive=True)

cars = []
notcars = []

for image in carsImages:
    cars.append(image)
for image in notcarsImages:
    notcars.append(image)

print(len(cars))
print(len(notcars))

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

# Define the labels vector
#X = [cars,notcars]
X = cars+notcars
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

### TODO: Tweak these parameters and see how the results change.
#colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

#hog_channels = [0,1,2,"ALL"]
#for hog_channel in hog_channels:
#    print(hog_channel)

#cell_per_blocks = [1,2,4,8]
#for cell_per_block in cell_per_blocks:
#    print(cell_per_block)
#
#pix_per_cells = [4,8,16,32]
#for pix_per_cell in pix_per_cells:
#    print(pix_per_cell)
    
#colorspaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
#for colorspace in colorspaces:
#    print(colorspace)
#orients = [3,5,7,9,11,13,15]
#for orient in orients:
#    print("orient=",orient)
colorspace = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

t=time.time()
feat_train = extract_features(X_train, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,feature_vec=True)
feat_test = extract_features(X_test, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,feature_vec=True)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
#X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
feat_scaler = StandardScaler().fit(feat_train)
# Apply the scaler to X
feat_train_scaled = feat_scaler.transform(feat_train)
feat_test_scaled = feat_scaler.transform(feat_test)


print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(feat_train_scaled, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(feat_test_scaled, y_test), 4))
# Check the prediction time for a single sample
#t=time.time()
#n_predict = 10
#print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
#t2 = time.time()
#print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')



#images = glob.glob('*.jpeg')
#cars = []
#notcars = []
#for image in images:
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)
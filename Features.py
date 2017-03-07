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


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    hog_image = None
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features, hog_image

def extract_features(image, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,feature_vec=False,vis=False):
    # Color space conversion
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_channel = [0,1,2]
        
    hog_features = []
    for channel in hog_channel:
        features,hog_image = get_hog_features(feature_image[:,:,channel], 
                            orient, pix_per_cell, cell_per_block, 
                            vis=vis, feature_vec=False)
        hog_features.append(features)
    hog_features = np.concatenate(hog_features,axis=4)
    if feature_vec:
        hog_features = np.ravel(hog_features)      
#    else:
#        features,_ = get_hog_features(feature_image[:,:,hog_channel], orient, 
#                    pix_per_cell, cell_per_block, vis=vis, feature_vec=feature_vec)
#        hog_features = features
    if vis==True:
        return hog_features,hog_image
    else:
        return hog_features

# Define a function to extract features from a list of images
def extract_features_from_image_list(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,feature_vec=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        
        # Extract HOG features
        hog_features = extract_features(image, cspace=cspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,feature_vec=feature_vec)
        
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features
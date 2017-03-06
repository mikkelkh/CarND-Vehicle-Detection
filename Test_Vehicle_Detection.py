#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:36:26 2017

@author: mikkel
"""

import numpy as np
import cv2
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.misc import imsave
import pickle
import os
import glob
from Features import *
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label

## Define helper functions ##

# This function is taken directly from Udacity sample code
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
 
# This function is taken directly from Udacity sample code
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# This function is taken directly from Udacity sample code
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# This function is taken directly from Udacity sample code
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Single function extracting features using HOG sub-sampling and making predictions
# This function is adapted from Udacity sample code
def find_cars(img, ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc):
    # Convert from RGB color convention to BGR (OpenCV)
    img_tosearch = img[ystart:ystop,:,[2,1,0]]
    
    # Loop over requested scales
    for scale in scales:
        # Only rescale if scale is different from 1
        if scale != 1:
            imshape = img_tosearch.shape
            img_scaled_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        else:
            img_scaled_tosearch = img_tosearch
    
        # Define blocks and steps to do HOG sub-sampling
        nxblocks = (img_scaled_tosearch.shape[1] // pix_per_cell)-1
        nyblocks = (img_scaled_tosearch.shape[0] // pix_per_cell)-1 

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute all HOG features in one go for the entire image
        hog = extract_features(img_scaled_tosearch, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel,feature_vec=False)
        
        # Instantiate an empty list that all detected cars (bounding boxes) can be put into
        window_list = []
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract the relevant part of the "HOG image"                
                hog_features = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
    
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Extract the image patch (only for visualization)
                subimg = cv2.resize(img_scaled_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
    
                # Scale features
                test_features = feat_scaler.transform(hog_features.ravel().reshape(1, -1))
                
                # Make prediction
                test_prediction = svc.predict(test_features)
                
                # Add bounding box to windows_list if it's a car
                if test_prediction[0] == 1:
    #                plt.imshow(subimg)
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    window_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    # Return all detections as bounding boxes
    return window_list

####################

# Load trained model and its HOG parameters
svc_pickle = pickle.load( open("svc_pickle.p", "rb" ) )

colorspace = svc_pickle['colorspace']
orient = svc_pickle['orient']
pix_per_cell = svc_pickle['pix_per_cell']
cell_per_block= svc_pickle['cell_per_block']
hog_channel = svc_pickle['hog_channel']
feat_scaler = svc_pickle['feat_scaler']
svc = svc_pickle['svc']

# Window search parameters
ystart = 400
ystop = 656
scales = [1.5,1]

# Processing pipeline
def process_image(self,image):
    # Extract features, perform sliding window and make predictions
    windows = find_cars(image, ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc)
    
    # Tracking
    # FIFO buffer with previously detected bounding boxes
    if len(self.bboxes)>1:
        self.bboxes = (self.bboxes[1:]+[windows]).copy()
    else:
        self.bboxes = [windows]
        
    # Instantiate empty heatmap 
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add all bounding boxes from this frame and previous frames (FIFO buffer)
    for bbox in self.bboxes:
        heat = add_heat(heat,bbox)

    # Apply threshold to help remove false positives (depends on FIFO buffer length)
    heat = apply_threshold(heat,1+len(self.bboxes)*2/3)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    # Draw bounding boxes onto image
    window_img = draw_labeled_bboxes(image, labels)
    
#    color_warp = np.dstack((heat, heat, heat))
    return window_img

# Define processing class. This is used to hold the FIFO buffer that is used for video processing
class ProcessClass:
    def __init__(self, nFrames):
        # Instantiate empty FIFO buffer with length 'nFrames'
        self.bboxes = ([[]]*nFrames).copy()
    def process_frame(self, frame):
        # process frame
        return process_image(self,frame)





## Run algorithm on: Test images
files = glob.glob('test_images/*.jpg')

# Intantiate processing class object with 1 frame FIFO buffer (single image)
processObj = ProcessClass(1)
for fname in files:
    # Load image
    image = cv2.imread(fname)
   
    # Process image
    window_img = processObj.process_frame(image[:,:,[2,1,0]].copy())

    imsave('output_images/' + fname,window_img)





## Run algorithm on: Video
from moviepy.editor import VideoFileClip

# Intantiate processing class object with 10 frames FIFO buffer (for video)
processObj = ProcessClass(10)

# Open video and process each frame
output = 'project_video_result.mp4'
clip1 = VideoFileClip("test_video.mp4")
clip1 = VideoFileClip("project_video.mp4")#.subclip(23,26)
white_clip = clip1.fl_image(processObj.process_frame)
white_clip.write_videofile(output, audio=False)
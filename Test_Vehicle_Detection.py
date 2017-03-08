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
from Detection import *
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label


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
xstart = 650
xstop = 1250
ystart = 400
ystop = 500
scales = [2.4,1.8,1.4,1]

## Run algorithm on: Test images
files = glob.glob('test_images/*.jpg')

# Intantiate processing class object with 1 frame FIFO buffer (single image)
processObj = ProcessClass(1, xstart, xstop, ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc)
for fname in files:
    # Load image
    image = cv2.imread(fname)
   
    # Process image
    window_img = processObj.process_frame(image[:,:,[2,1,0]].copy())

    imsave('output_images/' + fname,window_img)





## Run algorithm on: Video
from moviepy.editor import VideoFileClip

# Intantiate processing class object with 10 frames FIFO buffer (for video)
processObj = ProcessClass(20, xstart, xstop, ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc)

# Open video and process each frame
output = 'project_video_result.mp4'
#clip1 = VideoFileClip("test_video.mp4")
clip1 = VideoFileClip("project_video.mp4")#.subclip(16,20)
white_clip = clip1.fl_image(processObj.process_frame)
white_clip.write_videofile(output, audio=False)
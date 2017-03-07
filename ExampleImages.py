#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:02:10 2017

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
from Detection import *
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.misc import imsave
from scipy.ndimage.measurements import label



## Show random car and non-car ##
srcFolder = '../data/CarND-Vehicle-Detection'
cars = glob.glob(os.path.join(srcFolder,'vehicles','**/*.png'),recursive=True)
notcars = glob.glob(os.path.join(srcFolder,'non-vehicles','**/*.png'),recursive=True)

car = mpimg.imread(cars[0])
notcar = mpimg.imread(notcars[0])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.tight_layout()
ax1.imshow(car)
ax1.set_title('Car', fontsize=30)
ax2.imshow(notcar)
ax2.set_title('Not car', fontsize=30)
extent = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
plt.savefig('output_images/car_notcar.jpg', bbox_inches=extent.expanded(1, 1.2), dpi=50)




## Show example of HOG images ##
hog_features,hog_car = extract_features(car, cspace='YCrCb', orient=9, pix_per_cell=8, 
                                cell_per_block=2, hog_channel="ALL",feature_vec=False,vis=True)
hog_features,hog_notcar = extract_features(notcar, cspace='YCrCb', orient=9, pix_per_cell=8, 
                                cell_per_block=2, hog_channel="ALL",feature_vec=False,vis=True)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.tight_layout()
ax1.imshow(hog_car)
ax1.set_title('HOG image (car)', fontsize=30)
ax2.imshow(hog_notcar)
ax2.set_title('HOG image (Not car)', fontsize=30)
extent = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
plt.savefig('output_images/hog.jpg', bbox_inches=extent.expanded(1, 1.2), dpi=50)




## Show detection examples ##
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

file = 'test_images/test6.jpg'
image1 = cv2.imread(file)
windows1 = find_cars(image1[:,:,[2,1,0]], ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc)
car_with_boxes1 = draw_boxes(image1[:,:,[2,1,0]].copy(), windows1, color=(0, 0, 255), thick=6)

file = 'test_images/test2.jpg'
image2 = cv2.imread(file)
windows2 = find_cars(image2[:,:,[2,1,0]], ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc)
car_with_boxes2 = draw_boxes(image2[:,:,[2,1,0]].copy(), windows2, color=(0, 0, 255), thick=6)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
f.tight_layout()
ax1.imshow(car_with_boxes1)
ax1.set_title('test6.jpg', fontsize=30)
ax2.imshow(car_with_boxes2)
ax2.set_title('test2.jpg', fontsize=30)
extent = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
plt.savefig('output_images/boxes.jpg', bbox_inches=extent.expanded(1, 1.2), dpi=50)




## Show heatmap example ##
heat = np.zeros_like(image1[:,:,0]).astype(np.float)
heat = add_heat(heat,windows1)
heat = apply_threshold(heat,1)    
heatmap = np.clip(heat, 0, 255)
labels = label(heatmap)
window_img = draw_labeled_bboxes(image1[:,:,[2,1,0]].copy(), labels)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
f.tight_layout()
ax1.imshow(heatmap,cmap="hot")
ax1.set_title('Heatmap', fontsize=30)
ax2.imshow(window_img)
ax2.set_title('Thresholded heatmap', fontsize=30)
extent = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
plt.savefig('output_images/heat.jpg', bbox_inches=extent.expanded(1, 1.2), dpi=50)

## Show heatmaps on video example ##
from moviepy.editor import VideoFileClip
processObj = ProcessClass(10, ystart, ystop, scales, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, feat_scaler, svc, debug=True)
clip1 = VideoFileClip("project_video.mp4").subclip(39,41)
white_clip = clip1.fl_image(processObj.process_frame)
white_clip.write_videofile('test.mp4', audio=False)
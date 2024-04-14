# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:28:42 2024

@author: Hakkefar
"""
#last frame?
#contur av
#prøv annen thresholding enn otsu

import cv2
import numpy as np
import sys

# Resize the video window
cv2.namedWindow('GrayANDbwVideo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('GrayANDbwVideo', 800, 400)  # Adjust the width and height as needed
cv2.namedWindow('ColorVideo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ColorVideo', 800, 400)  # Adjust the width and height as needed

# Open the video file
cap = cv2.VideoCapture('fish.mp4')

threshvalue = 125
# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the frame position to the last frame
comparing_frame = total_frames - 8
cap.set(cv2.CAP_PROP_POS_FRAMES, comparing_frame)

# Read the first frame
ret, firstframe = cap.read()
if not ret:
    print("Error reading video file")
    exit()
gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

#semi_denoised_frame = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
#denoised_frame = cv2.medianBlur(semi_denoised_frame, 5)  # Adjust the kernel size (5x5) as needed
#blur2 = cv2.GaussianBlur(denoised_frame, (5, 5), 0)

# Perform adaptive thresholding
fthresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 5)
ret, fthresh3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret, fthresh4 = cv2.threshold(blur,threshvalue,255,cv2.THRESH_BINARY)
lastframe = cv2.bitwise_not(fthresh4)

# Get the height and width of the video frames
height, width = firstframe.shape[:2]


cap.set(cv2.CAP_PROP_POS_FRAMES, 1)


while True:
    # Read a new frame
    ret, frame = cap.read()

    if not ret:
        break

    #in case current frame is the same as the frame we are comparing to. 
    #Code will fail as everything will be removed and nothing left to analyse.    
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #get the number of current frame
    if current_frame == comparing_frame:
        break
        #print(current_frame,  " ",  comparing_frame)
        #ret, frame = cap.read()
        #print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        #if not ret:
        #    break
        
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #semi_denoised_frame = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    #denoised_frame = cv2.medianBlur(semi_denoised_frame, 5)  # Adjust the kernel size (5x5) as needed
    #blur2 = cv2.GaussianBlur(denoised_frame, (5, 5), 0)

    # Perform different thresholdings
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 5)
    ret, thresh3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, thresh4 = cv2.threshold(blur,threshvalue,255,cv2.THRESH_BINARY)
    currentframe = cv2.bitwise_not(thresh4)
        
    fish_only = cv2.bitwise_xor(thresh4, fthresh4)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erode_frame = cv2.erode(fish_only,kernel,8)
    dilate_frame = cv2.dilate(erode_frame, kernel,10)
    blur2 = cv2.GaussianBlur(dilate_frame, (7, 7), 0)

    contours, hierarchy = cv2.findContours(blur2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    # Create a new mask for the result image
    #h, w = cap.shape[:2]
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width
    mask = np.zeros((h, w), np.uint8)
    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    #res_frame = cv2.bitwise_and(frame, frame, mask=mask)
    if mask.shape[:2] != frame.shape[:2]:
        print("Error: Mask and frame dimensions mismatch")
        sys.exit()

    # Perform the bitwise operation only if the mask and frame have the same size and data type
    #if mask.dtype == np.uint8 and mask.dtype == frame.dtype:
    # Resize the mask to match the dimensions of the frame
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Perform the bitwise operation
    res_frame = cv2.bitwise_and(frame, frame, mask=mask_resized)
    res_gray = cv2.cvtColor(res_frame, cv2.COLOR_BGR2GRAY)    
    
    cv2.imshow('ColorVideo', res_frame)
    line1 = np.concatenate((dilate_frame, erode_frame,currentframe), axis=1)
    line2 = np.concatenate((lastframe, fish_only, res_gray), axis=1)
    mosaic = np.concatenate((line1, line2), axis=0)

    cv2.imshow('GrayANDbwVideo', mosaic)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

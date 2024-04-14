#!/usr/bin/env python3

import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('fish.mp4')

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file 'fish.mp4'")
    exit()
    

frame_count = 0  # To control how many images of the array to show

# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Only show every nth image in array
    if frame_count % 5 != 0:
        continue
    
    # Preprocessing by resizing and grayscaling
    resized_frame = cv2.resize(frame, (224, 224))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Denoising using 
    semi_denoised_frame = cv2.fastNlMeansDenoising(gray_frame, None, 10, 7, 21)
    denoised_frame = cv2.medianBlur(semi_denoised_frame, 5)  # Adjust the kernel size (5x5) as needed

    # Mirror for data augmentation
    mirrored_denoised_frame = cv2.flip(denoised_frame, 1)


    # Display original and denoised frames
    mosaic = np.concatenate((gray_frame, denoised_frame, mirrored_denoised_frame), axis=1)
    

    cv2.imshow('Video', mosaic)
    
    key = cv2.waitKey(1)  # Set playback speed (ms between frames)
    if key == ord('q'):    # Add option to exit program
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
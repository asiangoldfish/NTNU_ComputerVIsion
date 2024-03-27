import cv2      # Using functions from OpenCV
import numpy as np
import sys

erode_value = 1
filter_value = -3

cap = cv2.VideoCapture('fish.mp4')

if not cap.isOpened():
    print("Error: Could not open video file 'fish.mp4'")
    exit()
    
# Find total number of frames to find the last frame
# (for background extraction)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the video capture position to the last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

ret, last_frame = cap.read()  # Get the last 
if not ret:
    print("Error: Failed to read last frame from video")
    exit()
# Structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
 
# Resize and turn to grayscale to match frames in loop 
#resized_last_frame = cv2.resize(last_frame, (224, 224))
#normalized_last_frame = resized_last_frame.astype(float) / 255.0
#noiceremoval_last_frame = cv2.erode(resized_last_frame,kernel,erode_value)
#noiceremoval_last_frame = cv2.dilate(noiceremoval_last_frame, kernel,erode_value)
#noiceremoval_last_frame = cv2.filter2D(noiceremoval_last_frame, filter_value, kernel)
#gray_last_frame = cv2.cvtColor(noiceremoval_last_frame, cv2.COLOR_BGR2GRAY)
#blurred_last_frame = cv2.GaussianBlur(gray_last_frame, (5,5), 0)


# Resize the video window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)  # Adjust the width and height as needed

# Reset the video capture position to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)


# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Preprocessing
    #frame = cv2.fastNlMeansDenoising(frame,1)
    #frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    #resized_frame = cv2.resize(frame, (800, 600))
    #cap_grayscale = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernelLine = cv2.getStructuringElement(cv2.MORPH_RECT, (2,5))
    kernelLine2 = np.ones((1,10),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    kernel3 = np.ones((6,6),np.uint8)
 
    #color_fish = cv2.filter2D(frame, -1, kernel)
    gray_fish = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #normalized_frame = resized_frame.astype(float) / 255.0
    noiceremoval_frame = cv2.erode(gray_fish,kernel,erode_value)
    noiceremoval_frame = cv2.dilate(gray_fish, kernel,erode_value)
    #noiceremoval_frame = cv2.filter2D(noiceremoval_frame, filter_value, kernel)
    #gray_frame = cv2.cvtColor(noiceremoval_frame, cv2.COLOR_BGR2GRAY)
    #blurred_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    #gray_fish = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_fish,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Creating binary-picture and Removing boubles
    #kernel = np.zeros((2,2),np.uint8)
    opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel2, iterations = 2)
    bin_erodeLines = cv2.erode(opening,kernelLine2,iterations=15)
    bin_dilateLines = cv2.dilate(bin_erodeLines,kernelLine2,iterations=3)

    # Combine surrounding noise with ROI
    dilate = cv2.dilate(bin_dilateLines,kernel3,iterations=3)

    # Blur the image for smoother ROI
    blur = cv2.blur(bin_dilateLines,(15,15))

    # Perform another OTSU threshold and search for biggest contour
    ret, thresh2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
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
    # Display the resulting frame
    line1 = np.concatenate((thresh1, opening, bin_dilateLines, bin_erodeLines), axis=1)
    line2 = np.concatenate((dilate, blur, thresh2, res_gray), axis=1)
    mosaic = np.concatenate((line1,line2), axis=0)
    
    cv2.imshow('Video', mosaic)
    key = cv2.waitKey(1)  # Set playback speed (ms between frames)
    if key == ord('q'):    # Add option to exit program
        break

cap.release()
cv2.destroyAllWindows()

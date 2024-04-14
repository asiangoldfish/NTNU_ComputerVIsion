import cv2      # Using functions from OpenCV
import numpy as np

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
    
# Resize and turn to grayscale to match frames in loop 
resized_last_frame = cv2.resize(last_frame, (224, 224))
#normalized_last_frame = resized_last_frame.astype(float) / 255.0
gray_last_frame = cv2.cvtColor(resized_last_frame, cv2.COLOR_BGR2GRAY)
blurred_last_frame = cv2.GaussianBlur(gray_last_frame, (5,5), cv2.BORDER_DEFAULT)


# Resize the video window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)  # Adjust the width and height as needed

# Reset the video capture position to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    # if not ret:
    #     break

    # Preprocessing 
    resized_frame = cv2.resize(frame, (224, 224))
    #normalized_frame = resized_frame.astype(float) / 255.0

    # Convert to HSV
    # hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    #blurred_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
    gaussian_blurred_frame = cv2.GaussianBlur(gray_frame,(5,5),cv2.BORDER_DEFAULT)

    # Background subtraction
    bad_background_subtracted_frame = cv2.absdiff(gray_frame, gray_last_frame)
    background_subtracted_frame = gaussian_blurred_frame - blurred_last_frame

    # Canny edge detection
    edges = cv2.Canny(background_subtracted_frame, 100, 200)

    
    # Object detection
        # HOG
    
    # Shape/feature extraction
    

    # Thresholding
    _, binary_mask = cv2.threshold(background_subtracted_frame, 30, 255, cv2.THRESH_BINARY)
        # Background extraction, subtract the last frame
        # Morphological operations (erosion, dilation, opening, closing)
        
    # CNN
        # Kahi?
        
    # cv2.imshow('Video', binary_mask)
    # cv2.imshow('Video', resized_frame)
    # cv2.imshow('Video', gray_frame)
    # cv2.imshow('Video', gaussian_blurred_frame)
    # cv2.imshow('Video', background_subtracted_frame)
    # cv2.imshow('Video', edges)

    # line1 = np.concatenate((resized_frame, gray_frame, gaussian_blurred_frame), axis=1)
    line1 = np.concatenate((gray_frame, gaussian_blurred_frame, bad_background_subtracted_frame), axis=1)
    line2 = np.concatenate((background_subtracted_frame, edges, np.zeros_like(edges)), axis=1)
    mosaic = np.concatenate((line1, line2), axis=0)

    cv2.imshow('Video', mosaic)
    
    key = cv2.waitKey(25)  # Set playback speed (ms between frames)
    if key == ord('q'):    # Add option to exit program
        break

cap.release()
cv2.destroyAllWindows()

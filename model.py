import cv2


cap = cv2.VideoCapture('fish.mp4')

# Resize the video window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)  # Adjust the width and height as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # Preprocessing goes here
        # Noise reduction: Gaussian blurring and median filtering
        # Noramlizing and balancing colors, deblur motion blurring
    
    # Object detection
        # HOG
    
    # Shape extraction
        # Background extraction, subtract the last frame
    
    # CNN
        # Khai?
    
    

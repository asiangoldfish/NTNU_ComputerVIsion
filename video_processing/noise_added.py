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
    if frame_count % 1 != 0:
        continue
    
    # Resizing the frame
    resized_frame = cv2.resize(frame, (224, 224))

    # Adding Gaussian noise
    noise = np.random.normal(0, 1, resized_frame.size)
    noise = noise.reshape(resized_frame.shape).astype('uint8')
    
    noisy_frame = cv2.add(resized_frame, noise)

    # Write denoised frame to file
    cv2.imwrite(f'noise_increased/denoised_frame_{frame_count}.jpg', noisy_frame)

    # Display original and denoised frames
    #mosaic = np.concatenate((resized_frame, noisy_frame), axis=1)
    #cv2.imshow('Video', mosaic)
    
    key = cv2.waitKey(25)  # Set playback speed (ms between frames)
    if key == ord('q'):    # Add option to exit program
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
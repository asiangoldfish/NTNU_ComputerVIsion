import cv2
import numpy as np

#taken from stackoverflow
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

# Open the video file
cap = cv2.VideoCapture('fish.mp4')

cv2.namedWindow('ColorVideo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ColorVideo', 224*3, 224*2)  # Adjust the width and height as needed

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

    #HSV Not used to anything useful yet.
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    # Adding Gaussian noise
    noise = np.random.normal(0, 1, resized_frame.size)
    noise = noise.reshape(resized_frame.shape).astype('uint8')    
    noisy_frame = cv2.add(resized_frame, noise)

    # Denoising the frame 
    semi_denoised_frame = cv2.fastNlMeansDenoising(resized_frame, None, 10, 7, 21)
    denoised_frame = cv2.medianBlur(semi_denoised_frame, 5)  # Adjust the kernel size (5x5) as needed

    #change illumination
    darker_frame = adjust_gamma(resized_frame, gamma=0.4) #adjust gamma down for darker, and up for lighter. 1 = no change.
    lighter_frame = adjust_gamma(resized_frame, gamma=1.6)

    #display results
    line1 = np.concatenate((resized_frame, denoised_frame, noisy_frame), axis=1)
    line2 = np.concatenate((darker_frame, lighter_frame, hsv_frame), axis=1)
    mosaic = np.concatenate((line1, line2), axis=0)    
    cv2.imshow('ColorVideo', mosaic)
    
    # Write frames to files
    cv2.imwrite(f'noise_increased/noisy_frame_{frame_count}.jpg', noisy_frame)
    cv2.imwrite(f'noise_reduced/denoised_frame_{frame_count}.jpg', denoised_frame)
    cv2.imwrite(f'illumination_decreased/darker_frame_{frame_count}.jpg', darker_frame)
    cv2.imwrite(f'illumination_increased/lighter_frame_{frame_count}.jpg', lighter_frame)
    cv2.imwrite(f'hsv/hsv_frame_{frame_count}.jpg', hsv_frame)
      
    key = cv2.waitKey(25)  # Set playback speed (ms between frames)
    if key == ord('q'):    # Add option to exit program
        break

# Release the video capture object and close all windows
print("Done!")      #In case the imshow is deactivated
cap.release()
cv2.destroyAllWindows()
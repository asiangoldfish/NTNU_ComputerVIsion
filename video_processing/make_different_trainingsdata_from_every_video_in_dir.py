import cv2
import numpy as np
import os
import glob

#Variables to activate/deactivate

#Activate this to show video
showVideo = False

#activate this to create files
createFiles = True

#activate this to run program fast, but frames will be reduced to 244x244
fastSpeed = False

#Set to 1 means going trough every frame. set to 5 to only process every 5.th frame.
speed_multiplyer = 1

#Function that changes illumination in a frame. gamma<1. means darker.
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)


mp4files = []
for file in glob.glob("*.mp4"):
    mp4files.append(file)
    
#print(mp4files)    
total_videos = len(mp4files)
video_count = 0
for file in mp4files:
    video_count += 1
    filesplit = file.split(".")
    filename = filesplit[0]     #filename without .mp4 
    print(filename + " - video: " + str(video_count) + " / " + str(total_videos))

    key = cv2.waitKey(1)  # Set playback speed (ms between frames)
    if key == ord('q'):    # Add option to exit program
        break


#taken from stackoverflow

    # Open the video file
    
    #code for looping trough every video, but this lines replaces this for now.
    
    cap = cv2.VideoCapture(file)
    
    #if showVideo :
    cv2.namedWindow('ColorVideo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ColorVideo', 224*4, 224*2)  # Adjust the width and height as needed
    #cv2.namedWindow('Colornormal', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Colornormal', 224, 224)  # Adjust the width and height as needed
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file 'fish.mp4'")
        exit()    
    
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0  # To control how many images of the array to show
    
    if not os.path.exists(f"{filename}/noise_increased"):
        os.makedirs(f"{filename}/noise_increased")
        
    if not os.path.exists(f"{filename}/noise_reduced"):
        os.makedirs(f"{filename}/noise_reduced")
        
    if not os.path.exists(f"{filename}/illumination_increased"):
        os.makedirs(f"{filename}/illumination_increased")
        
    if not os.path.exists(f"{filename}/illumination_decreased"):
        os.makedirs(f"{filename}/illumination_decreased")
        
    if not os.path.exists(f"{filename}/contrasted"):
        os.makedirs(f"{filename}/contrasted")
        
    #checking if folder exists
    #if not os.exists("noise_increased/"):
    #    os.create("noice_increased/")
    
    #if not os.path.exists("mappenavn"):
    #    os.makedirs("mappenavn")
        
    # Loop through each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_num = str(frame_count).zfill(6)
    
        # Only show every nth image in array
        if frame_count % speed_multiplyer != 0:
            continue
        
        # Resizing the frame
        if fastSpeed:
            resized_frame = cv2.resize(frame, (224, 224))
        else :
            resized_frame = frame
    
        #HSV Not used to anything useful yet.
        #hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        #normalized_frame = cv2.normalize(resized_frame, None, 0.05, 1.3, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
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
        
        #img = cv2.imread('flower.jpg', 1)
    
        # converting to LAB color space
        lab = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
    
        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
    
        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))
        
        # Converting image from LAB Color model to BGR color spcae
        contrasted_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
        # Stacking the original image with the enhanced image
        result = np.hstack((resized_frame, contrasted_frame))
        #cv2.imshow('Result', result)
    
        print("file: " + file + " - video: " + str(video_count) + " / " + str(total_videos) + " - frame: " + str(frame_count) + " / " + str(total_frames))
        
        #display results
        if showVideo :
            line1 = np.concatenate((resized_frame, contrasted_frame, denoised_frame), axis=1)
            line2 = np.concatenate((darker_frame, lighter_frame, noisy_frame), axis=1)
            mosaic = np.concatenate((line1, line2), axis=0)    
            cv2.imshow('ColorVideo', mosaic)
            #cv2.imshow('Colornormal', normalized_frame)
    
    
        # Write frames to files
        if createFiles :    
            cv2.imwrite(f'{filename}/noise_increased/frame{frame_num}.jpg', noisy_frame)
            cv2.imwrite(f'{filename}/noise_reduced/frame{frame_num}.jpg', denoised_frame)
            cv2.imwrite(f'{filename}/illumination_decreased/frame{frame_num}.jpg', darker_frame)
            cv2.imwrite(f'{filename}/illumination_increased/frame{frame_num}.jpg', lighter_frame)
            cv2.imwrite(f'{filename}/contrasted/frame{frame_num}.jpg', contrasted_frame)
          
    # Release the video capture object and close all windows
    print(file + " done!")      
    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import glob
import time
import sys


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
for file in glob.glob("dataset/*.mp4"):
    mp4files.append(file)
    
#print(mp4files)    
total_videos = len(mp4files)
video_count = 0

# No vides were found
if total_videos == 0:
    print("No videos were found. Please create a directory 'dataset/' and load all your MP4 videos here.")
    sys.exit(1)

for file in mp4files:
    start_time_video = time.time()
    video_count += 1

    if os.name == 'nt':
        filesplit = file.split("\\")
    else:
        filesplit = file.split("/")


    filesplit = filesplit[-1].split(".")
    filename = filesplit[0]     #filename without .mp4 

    print(filename + ".mp4 - video: " + str(video_count) + "/" + str(total_videos))

    # Open the video file
    cap = cv2.VideoCapture(file)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{filename}.mp4'")
        continue
    
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0  # To control how many images of the array to show
    
    if not os.path.exists(f"output/{filename}/noise_increased"):
        os.makedirs(f"output/{filename}/noise_increased")
        
    if not os.path.exists(f"output/{filename}/noise_reduced"):
        os.makedirs(f"output/{filename}/noise_reduced")
        
    if not os.path.exists(f"output/{filename}/illumination_increased"):
        os.makedirs(f"output/{filename}/illumination_increased")
        
    if not os.path.exists(f"output/{filename}/illumination_decreased"):
        os.makedirs(f"output/{filename}/illumination_decreased")
        
    if not os.path.exists(f"output/{filename}/contrasted"):
        os.makedirs(f"output/{filename}/contrasted")

    total_seconds_elapsed = 0
        
    # Loop through each frame of the video
    while True:
        start_time_frame = time.time()

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
            
        #display results
        if showVideo :
            line1 = np.concatenate((resized_frame, contrasted_frame, denoised_frame), axis=1)
            line2 = np.concatenate((darker_frame, lighter_frame, noisy_frame), axis=1)
            mosaic = np.concatenate((line1, line2), axis=0)    
            cv2.imshow('ColorVideo', mosaic)
            #cv2.imshow('Colornormal', normalized_frame)
    
    
        # Write frames to files
        if createFiles :    
            cv2.imwrite(f'output/{filename}/noise_increased/frame{frame_num}.jpg', noisy_frame)
            cv2.imwrite(f'output/{filename}/noise_reduced/frame{frame_num}.jpg', denoised_frame)
            cv2.imwrite(f'output/{filename}/illumination_decreased/frame{frame_num}.jpg', darker_frame)
            cv2.imwrite(f'output/{filename}/illumination_increased/frame{frame_num}.jpg', lighter_frame)
            cv2.imwrite(f'output/{filename}/contrasted/frame{frame_num}.jpg', contrasted_frame)
        
        # Elapsed time
        seconds_per_frame = time.time() - start_time_frame # Total elapsed time in seconds
        total_seconds_elapsed += seconds_per_frame
        elapsed_minutes = total_seconds_elapsed // 60
        elapsed_hours = elapsed_minutes // 60
        elapsed_seconds = total_seconds_elapsed - (elapsed_hours * 3600 + elapsed_minutes * 60)

        # print("file: " + file + " - video: " + str(video_count) + " / " + str(total_videos) + " - frame: " + str(frame_count) + " / " + str(total_frames) + " - {total_time_frame}secs")

        
        # Print loading bar
        print(f"\tFrame {frame_count + 1} of {total_frames} ", end='|')

        # Number of signs to print for loading bar
        loading_bar = ((frame_count + 1) / total_frames) * 100

        for j in range(int(loading_bar)):
            print("=", end='')

        # Print empty signs for remaining frames to render
        for j in range(100 - int(loading_bar)):
            print(" ", end='')

        print(f"| {100 * (frame_count + 1) / total_frames:.1f}% | ", end='')
        print(f"Elapsed: {int(elapsed_hours)}hr {int(elapsed_minutes)}m {int(elapsed_seconds)}s | Frame time: {'{:.4f}'.format(seconds_per_frame)}s", end='\r')

          
    # Release the video capture object and close all windows
    total_time_video = time.time() - start_time_video
    print(file + " done in {total_time_video} secs")      
    cap.release()
    cv2.destroyAllWindows()
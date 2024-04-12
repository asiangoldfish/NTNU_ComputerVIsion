import cv2
import numpy as np
import os
import glob
import time
import sys

import utils

#activate this to create files
createFiles = True

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)


mp4files = list()
annotations = list()

"""
Create a dictionary that maps the video name to the corresponding annotation
Example:
{
    "8172831_fish.mp4": [
        frame_number
    ]
}
"""

video_annotation = {}

# Get all video files
for file in glob.glob("dataset/*.mp4"):
    file = file.split('/')[-1]
    mp4files.append(file)

# Get all annotation files
for a in glob.glob("annotations/*.xml"):
    annotations.append(a)
    
total_videos = len(mp4files)
video_count = 0

# No vides were found
if total_videos == 0:
    print("No videos were found. Please create a directory 'dataset/' and load all your MP4 videos here.")
    sys.exit(1)

# From each annotation get all the frames with bounding boxes that belong to a
# corresponding video. This ensures saves performance, so we don't process
# frames without fishes.
for ann_file in annotations:
    found_boxes = utils.parse_annotations(os.path.join(ann_file), 1)
    for box in found_boxes:
        frame_num = box['frame_num']
        source = box['source']

        if source not in mp4files:
            break

        try:
            video_annotation[source].append(frame_num)
        except Exception as e:
            video_annotation[source] = list()
            video_annotation[source].append(frame_num)

# Process videos
for video_count, file in enumerate(mp4files):
    file_path = os.path.join('dataset', file)

    try:
        if len(video_annotation[f"{file}.mp4"]) == 0:
            # Skip if the video has no annotations
            continue
    except:
        pass

    # Logging
    start_time_video = time.time()

    # Get the filename without extension
    name, extension = os.path.splitext(file)
    filename = name

    print(f'{file_path} - video: {str(video_count + 1)}/{str(total_videos)}')

    # Open the video file
    cap = cv2.VideoCapture(file_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{file_path}'")
        continue
    
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0  # To control how many images of the array to show

    total_seconds_elapsed = 0

    # Video has no annotations?
    try:
        video_annotation[f"{filename}.mp4"]
    except:
        print("This video has no annotations. Skipping...")
        continue
        
    # Loop through each frame of the video
    while True:
        start_time_frame = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        if frame_count not in video_annotation[f"{filename}.mp4"]:
            # If the frame does not have annotation, then skip
            continue
        

        frame_num = str(frame_count).zfill(6)
    
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
    
        # Write frames to files
        cv2.imwrite(f'output/{filename}_noiseIncreased_frame{frame_num}.jpg', noisy_frame)
        cv2.imwrite(f'output/{filename}_noiseReduced_frame{frame_num}.jpg', denoised_frame)
        cv2.imwrite(f'output/{filename}_illuminationReduced_frame{frame_num}.jpg', darker_frame)
        cv2.imwrite(f'output/{filename}_illuminationIncreased_frame{frame_num}.jpg', lighter_frame)
        cv2.imwrite(f'output/{filename}_contrasted_frame{frame_num}.jpg', contrasted_frame)
                    
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

# Split the training set
utils.split_dataset()
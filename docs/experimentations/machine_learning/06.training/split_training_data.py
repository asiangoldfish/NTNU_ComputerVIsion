#!/usr/bin/env python3

import os
import glob
import shutil

def split_dataset() -> bool:
    # Exit if there is no output frames to split
    if not os.path.exists('output'):
        print("Cannot split the data into training and testing. There is no 'output' directory")
        return False


    frames = os.listdir("output")
    frames_count = len(frames)

    # No frames were found in output
    if frames_count == 0:
        print("No frames in 'output/'")
        return False

    train_count = int(frames_count * 0.8)
    test_count = frames_count - train_count

    images = list()
    annotations = list()

    # Get all images
    for file in glob.glob("output/*.jpg"):
        file = file.split('/')
        images.append(file[-1])

    # Get all annotations
    for file in glob.glob("output/*.txt"):
        file = file.split('/')
        annotations.append(file[-1])

    if not os.path.exists('output/train'):
        os.mkdir('output/train')

    if not os.path.exists('output/test'):
        os.mkdir('output/test')

    if len(annotations) == 0 and len(images) == 0:
        print("No images or files to split")
        return False

    # Split the frames into training and testing: 80/20
    for i, image in enumerate(images):
        if i < train_count:
            target = 'train'
        else:
            target = 'test'

        shutil.move(f'output/{image}', f'output/{target}/{image}', copy_function = shutil.copy2)
        

    # Split the annotations into training and testing: 80/20
    for i, annotation in enumerate(annotations):
        if i < train_count:
            target = 'train'
        else:
            target = 'test'

        shutil.move(f'output/{annotation}', f'output/{target}/{annotation}', copy_function = shutil.copy2)

    return True
        

if "__main__" == __name__:
    split_dataset()
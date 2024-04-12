#!/usr/bin/env python3

import os
import glob
import shutil
import xml.etree.ElementTree as ET

def parse_annotations(annotations_path, skip_frames):
    """
    Parse annotations from an XML file.

    Args:
        annotations_path (str): The path to the XML file containing annotations.
        skip_frames (int): The number of frames to skip.

    Returns:
        list: A list of tuples containing the source, frame number, and bounding box coordinates.
    """

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    annotations = list()    # A new `annotations` tuple for the current xml file
    source = root.find('meta').find('source').text

    for track in root.findall('track'):
        for idx, box in enumerate(track):
            if idx % skip_frames != 0:    # if the annotation doesn't match the given interval
                continue                    # skip the frame
            frame_nr = int(box.get('frame'))
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            annotations.append({
                "source": source,
                "frame_num": frame_nr,
                "xtl": xtl,
                "ytl": ytl,
                "xbr": xbr, 
                "ybr": ybr
            })
    return annotations


def split_dataset() -> bool:
    # Exit if there is no output frames to split
    if not os.path.exists('output'):
        print("Cannot split the data into training and testing. There is no 'output' directory")
        return False


    frames = glob.glob('output/*.jpg')
    # Count frames
    frames_count = len(frames)

    # No frames were found in output
    if frames_count == 0:
        print("No frames in 'output/'")
        return False

    train_count = int(frames_count * 0.8)

    images = list()
    annotations = list()

    # Get all images
    for file in frames:
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
    for i, file in enumerate(images):
        if i < train_count:
            target = 'train'
        else:
            target = 'test'

        name, _ = os.path.splitext(file)

        # Move image
        shutil.move(f'output/{name}.jpg', f'output/{target}/{name}.jpg', copy_function = shutil.copy2)

        # Move annotation
        shutil.move(f'output/{name}.txt', f'output/{target}/{name}.txt', copy_function = shutil.copy2)
        

    # Split the annotations into training and testing: 80/20
    # for i, file in enumerate(annotations):
    #     if i < train_count:
    #         target = 'train'
    #     else:
    #         target = 'test'

    #     shutil.move(f'output/{file}', f'output/{target}/{file}', copy_function = shutil.copy2)

    return True

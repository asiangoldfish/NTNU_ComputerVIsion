#!/usr/bin/env python3

import argparse
import configparser
import os
import cv2
import numpy as np
import time
import concurrent.futures
import xml.etree.ElementTree as ET
# import progressbar
from alive_progress import alive_bar
import albumentations as A

parser = argparse.ArgumentParser()
parser.add_argument(
    '--visual',
    action='store_true',
    help='Show processed images on screen.'
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='Increase output verbosity instead of showing progress bar.'
)
parser.add_argument(
    '--no-create-files',
    action='store_true',
    help='Skip saving processed images to disk.'
)
parser.add_argument(
    '--reduced-res',
    action='store_true',
    help='Process images at reduced resolution.'
)
parser.add_argument(
    '--skip-frames',
    type=int,
    default=1,
    help='Interval at which frames should be processed.'
)
parser.add_argument(
    '--threads',
    type=int,
    default=0,
    help='Number of threads to use. 0 = use all available.'
)
parser.add_argument(
    '--input-dir',
    type=str,
    default='../ANADROM/Annotation',
    help='Input directory containing videos.'
)
parser.add_argument(
    '--output-dir',
    type=str,
    default='../ANADROM/training_data/',
    help='Output directory for processed videos.'
)
parser.add_argument(
    '--config',
    type=str,
    default='video_processing/cfg/get_augment_annotate.cfg',
    help='Path to config file specifying which data augmentation to perform.'
)

args = parser.parse_args()

# pool to add threads to process and augment frames concurrently
if args.threads == 0:
    pool = concurrent.futures.ThreadPoolExecutor()
else:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.threads)

# read config file
config = configparser.ConfigParser(allow_no_value=True)
config.read(args.config)

# augments = config['augments']
augments = [key for key in config['augments']]
num_augments = len(augments)
if args.verbose:
    print("The following augments will be applied: {}".format(augments))

# # set up progress bar
# bar_widgets = [' [',
#                progressbar.Timer(format = 'elapsed time: %(elapsed)s'),
#                '] ',
#                 progressbar.Bar('*'),' (',
#                 progressbar.ETA(), ') ',
#                 ]
# # progress bar is accessible globally
# bar = progressbar.ProgressBar()

def main():
    # print out warnings
    if args.no_create_files:
        print("CAUTION: Will not save processed images to disk.")
        print("CAUTION: Will not save processed images to disk.")
        print("CAUTION: Will not save processed images to disk.")
    if args.verbose:
        print("Output verbosity increased.")
        if args.visual:
            print("Showing processed images on screen. This will slow down processing significantly.")
        if args.reduced_res:
            print("Processing images at reduced resolution.")
        if args.threads != 0:
            print("Processing limited to {} threads.".format(args.threads))

    # get input and output directories
    # input_dir = os.path.abspath(args.input_dir)
    # output_dir = os.path.abspath(args.output_dir)
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not args.no_create_files:
        os.makedirs(args.output_dir, exist_ok=True)

    process_video_annotations(input_dir, output_dir)

def process_video_annotations(input_dir, output_dir):
    """
    Process annotations for all videos in a directory.

    Args:
        input_dir (str): The directory containing video annotations.
        output_dir (str): The directory to save the cropped frames.
    """

    with alive_bar(len(os.listdir(input_dir))) as bar:
        for video in os.listdir(input_dir):

            #debug
            if args.verbose:
                print('Processing video {}'.format(video))

            # skip non-video directories
            if not os.path.isdir(os.path.join(input_dir, video)):
                #debug
                if args.verbose:
                    print('Processing video: Skipping non-video directory {}'.format(video))

                continue
            annotations_path = os.path.join(input_dir, video, 'annotations.xml')
            frames_dir = os.path.join(input_dir, video, 'images')

            #debug
            if args.verbose:
                print('Processing: annotations path: ', annotations_path)
                print('Processing: frames path: ', frames_dir)

            augment_video(annotations_path, frames_dir, output_dir)
            bar()
        # pool.submit(augment, annotations_path, frames_dir, output_dir)

    # wait for all threads to finish
    pool.shutdown(wait=True)

def augment_video(annotations_path, frames_dir, output_dir):
    """
    Augment frames and pair them with YOLO-compatible annotations.

    Args:
        annotations_path (str): The path to the XML file containing annotations.
        frames_dir (str): The directory containing the frames.
        output_dir (str): The directory to save the cropped frames.
    """

    if args.verbose:
        print("Augmenting video {}".format(annotations_path))

    annotations = parse_annotations(annotations_path, args.skip_frames)
    if len(annotations) == 0:
        #debug
        if args.verbose:
            print("No annotations found in {}".format(annotations_path))

        return

    # start the progress bar
    # if not args.verbose:
        # bar.start(len(annotations))
        # bar.widgets = bar_widgets
        # bar.max_value = len(annotations)
        # bar.start()
        # bar = progressbar.ProgressBar(widgets=bar_widgets, max_value=len(annotations)).start()
    # if not args.verbose:model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        # bar = progressbar.ProgressBar(widgets=bar_widgets, max_value=len(annotations)).start()
        # bar = progressbar.ProgressBar(widgets = bar_widgets).start(max_value = len(annotations))

    # `idx` is needed to differentiate the names of multiple bounding boxes in the same frame
    for idx, (source, frame_nr, xtl, ytl, xbr, ybr) in enumerate(annotations, start=1):
        x_centre, y_centre, w, h = bbox_to_yolo(xtl, ytl, xbr, ybr)

        if args.verbose:
            print("Starting thread {} of {}".format(idx, len(annotations)))

        pool.submit(
            augment_and_annotate,
            frames_dir, output_dir,
            idx, source, frame_nr, x_centre, y_centre, w, h
        )

    # wait for all threads to finish
    # pool.shutdown(wait=True)

def parse_annotations(annotations_path, skip_frames):
    """
    Parse annotations from an XML file.

    Args:
        annotations_path (str): The path to the XML file containing annotations.
        skip_frames (int): The number of frames to skip.

    Returns:
        list: A list of tuples containing the source, frame number, and bounding box coordinates.
    """

    if args.verbose:
        print("Parsing annotations from {}".format(annotations_path))

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    annotations = ()    # A new `annotations` tuple for the current xml file
    source = root.find('meta').find('source').text
    if args.verbose:
        print("Source: {}".format(source))

    for track in root.findall('track'):
        for idx, box in enumerate(track):
            if idx % skip_frames != 0:    # if the annotation doesn't match the given interval
                continue                    # skip the frame
            frame_nr = int(box.get('frame'))
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            annotations = annotations + ((source, frame_nr, xtl, ytl, xbr, ybr),)
            if args.verbose:
                print("Found bbox {} {} {} {} in {}".format(xtl, ytl, xbr, ybr, frame_nr))
    if len(annotations) == 0:
        if args.verbose:
            print("No annotations found in {}".format(annotations_path))
    return annotations
        
def augment_and_annotate(frames_dir, output_dir, idx, source, frame_nr, x_centre, y_centre, w, h):
    frame_path = os.path.join(frames_dir, 'frame_{:06d}.PNG'.format(frame_nr))

    if args.verbose:
        print("Augmenting frame {} from {}".format(frame_nr, frame_path))

    frame = cv2.imread(frame_path)
    if args.visual:
        cv2.imshow('Frame', frame)
    if args.verbose:
        print("Read frame {} from {}".format(frame_nr, frame_path))

    if args.reduced_res:
        frame = cv2.resize(frame, (224, 224))
        if args.verbose:
            print("Resized frame {} from {}".format(frame_nr, frame_path))

    # compose output path for the new image
    output_path = os.path.join(output_dir, '{}_frame_{:06d}_{}.PNG'.format(source, frame_nr, idx))
    file_name, file_extension = os.path.splitext(output_path)

    if args.verbose:
        print("Output path: {}".format(output_path))
        print("File name: {}".format(file_name))

    bbox_orig = [0, x_centre, y_centre, w, h]
    # data augmentation: create transformed versions of the same image
    # we add every oriented version of the image here,
    # later we iterate through the list and apply all other augmentation to every orientation
    affine_transforms = []
    affine_transforms.append(('noflip', {
        'image': frame,
        'bboxes': [bbox_orig]
    }))
    if not args.verbose:
        print("Added noflip {} from {}".format(frame_nr, frame_path))
    if args.visual:
        cv2.imshow('noflip', frame)
        cv2.waitKey(0)

    #debug
    # checking contents of `augments`
    # if "flip_h" in augments:
    #     print("flip_h in augments")
    # if 'rotation' in augments:
    #     print("rotation in augments")
    # print('first element of augments: ', augments[0])

    if 'flip_h' in augments:
        # augments.remove('flip_h')
        transform_flip_h = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params = A.BboxParams(format = 'yolo'))
        transformed_flip_h = transform_flip_h(image=frame, bboxes=[bbox_orig])
        affine_transforms.append(('flip_h', transformed_flip_h))
        if args.verbose:
            print("Added flip_h {} from {}".format(frame_nr, frame_path))
        if args.visual:
            cv2.imshow('flip_h', transformed_flip_h['image'])
            cv2.waitKey(0)

    if 'flip_v' in augments:
        # augments.remove('flip_v')
        transform_flip_v = A.Compose([
            A.VerticalFlip(p=1),
        ], bbox_params = A.BboxParams(format = 'yolo'))
        transformed_flip_v = transform_flip_v(image=frame, bboxes=[bbox_orig])
        affine_transforms.append(('flip_v', transformed_flip_v))
        if args.verbose:
            print("Added flip_v {} from {}".format(frame_nr, frame_path))
        if args.visual:
            cv2.imshow('flip_v', transformed_flip_v['image'])
            cv2.waitKey(0)

    if 'flip_h_v' in augments:
        # augments.remove('flip_h_v')
        transform_flip_h_v = A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ], bbox_params = A.BboxParams(format = 'yolo'))
        transformed_flip_h_v = transform_flip_h_v(image=frame, bboxes=[bbox_orig])
        affine_transforms.append(('flip_h_v', transformed_flip_h_v))
        if args.verbose:
            print("Added flip_h_v {} from {}".format(frame_nr, frame_path))
        if args.visual:
            cv2.imshow('flip_h_v', transformed_flip_h_v['image'])
            cv2.waitKey(0)

    if 'rotation' in augments:
        transform_rotation = A.Compose([
            A.Affine(rotate=random.randint(0, 360), p=1),
        ], bbox_params = A.BboxParams(format = 'yolo'))
        transformed_rotation = transform_rotation(
            image=flipped[1]['image'],
            bboxes=flipped[1]['bboxes']
        )
        affine_transforms.append(('rotation', transformed_rotation))
        if args.verbose:
            print("Added rotation {} from {}".format(frame_nr, frame_path))
        if args.visual:
            cv2.imshow('rotation', transformed_rotation['image'])
            cv2.waitKey(0)

    if 'shear' in augments:
        transform_shear = A.Compose([
            A.Affine(shear=random.randint(-10, 10), p=1),
        ], bbox_params = A.BboxParams(format = 'yolo'))
        transformed_shear = transform_shear(
            image=flipped[1]['image'],
            bboxes=flipped[1]['bboxes']
        )
        affine_transforms.append(('shear', transformed_shear))
        if args.verbose:
            print("Added shear {} from {}".format(frame_nr, frame_path))
        if args.visual:
            cv2.imshow('shear', transformed_shear['image'])
            cv2.waitKey(0)

    # if the `augment_flipped` flag is set, loop through all the flipped versions of the frame
    # otherwise, only get the first element in the list
    for flipped in affine_transforms[0:] if augment_affine in augments else affine_transforms[:1]:
        augmented_frames = []

        # if 'rotation' in augments:
        #     transform_rotation = A.Compose([
        #         A.Affine(rotate=random.randint(0, 360), p=1),
        #     ], bbox_params = A.BboxParams(format = 'yolo'))
        #     transformed_rotation = transform_rotation(
        #         image=flipped[1]['image'],
        #         bboxes=flipped[1]['bboxes']
        #     )
        #     augmented_frames.append(('rotation', transformed_rotation))
        #     if not args.verbose:
        #         print("Added rotation {} from {}".format(frame_nr, frame_path))
        #     if args.visual:
        #         cv2.imshow('rotation', transformed_rotation['image'])
        #         cv2.waitKey(0)
        #
        # if 'shear' in augments:
        #     transform_shear = A.Compose([
        #         A.Affine(shear=random.randint(-10, 10), p=1),
        #     ], bbox_params = A.BboxParams(format = 'yolo'))
        #     transformed_shear = transform_shear(
        #         image=flipped[1]['image'],
        #         bboxes=flipped[1]['bboxes']
        #     )
        #     augmented_frames.append(('shear', transformed_shear))

        if 'lum_plus' in augments:
            augmented_frames.append(('lum_plus', {
                'image': adjust_gamma(flipped[1]['image'], gamma = 1.6),
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added lum_plus {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('lum_plus', flipped[1]['image'])
                cv2.waitKey(0)

        if 'lum_minus' in augments:
            augmented_frames.append(('lum_minus', {
                'image': adjust_gamma(flipped[1]['image'], gamma = 0.4),
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added lum_minus {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('lum_minus', flipped[1]['image'])
                cv2.waitKey(0)

        if 'noise_plus' in augments:
            noise = np.random.normal(0, 1, flipped[1]['image'].size)
            noise = noise.reshape(flipped[1]['image'].shape).astype('uint8')
            augmented_frames.append(('noise_plus', {
                'image': cv2.add(flipped[1]['image'], noise),
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added noise_plus {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('noise_plus', flipped[1]['image'])
                cv2.waitKey(0)

        if 'noise_minus' in augments:
            semi_denoised = cv2.fastNlMeansDenoisingColored(flipped[1]['image'], None, 10, 10, 7, 21)
            augmented_frames.append(('noise_minus', {
                'image': cv2.medianBlur(semi_denoised, 5),
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added noise_minus {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('noise_minus', flipped[1]['image'])
                cv2.waitKey(0)

        if 'contrast' in augments:
            lab = cv2.cvtColor(flipped[1]['image'], cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            # apply CLAHE to lightness
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            # merge the CLAHE enhanced L channel with the a and b channel
            limg = cv2.merge((cl, a, b))
            # Converting image from LAB Color model to RGB model
            contrasted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # stack original with contrasted
            augmented_frames.append(('contrast', {
                'image': np.hstack((flipped[1]['image'], contrasted)),
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added contrast {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('contrast', np.hstack((flipped[1]['image'], contrasted)))
                cv2.waitKey(0)

        if 'blue_tint' in augments:
            blue_tint_frame = np.copy(flipped[1]['image'])
            blue_tint_frame[:, :, 0] = cv2.add(blue_tint_frame[:, :, 0], 100)
            blue_tint_frame[:, :, 1] = cv2.add(blue_tint_frame[:, :, 1], 25)
            blue_tint_frame[:, :, 2] = cv2.add(blue_tint_frame[:, :, 2], 25)
            augmented_frames.append(('blue_tint', {
                'image': blue_tint_frame,
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added blue_tint {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('blue_tint', blue_tint_frame)
                cv2.waitKey(0)

        if 'green_tint' in augments:
            green_tint_frame = np.copy(flipped[1]['image'])
            green_tint_frame[:, :, 0] = cv2.add(green_tint_frame[:, :, 0], 25)
            green_tint_frame[:, :, 1] = cv2.add(green_tint_frame[:, :, 1], 50)
            augmented_frames.append(('green_tint', {
                'image': green_tint_frame,
                'bboxes': flipped[1]['bboxes']
            }))
            if args.verbose:
                print("Added green_tint {} from {}".format(frame_nr, frame_path))
            if args.visual:
                cv2.imshow('green_tint', green_tint_frame)
                cv2.waitKey(0)

        # save all to disk
        if not args.no_create_files:
            for frame in augmented_frames:
                frame_path = os.path.join(file_name, frame[0], file_extension)
                if args.verbose:
                    print("Saving file {}".format(frame_path))
                cv2.imwrite(frame_path, frame[1]['image'])
                annotations_path = os.path.join(file_name, frame[0], '.txt')
                with open(annotations_path, 'a') as f:
                    for bbox in frame[1]['bboxes']:
                        x_centre, y_centre, w, h = bbox
                        f.write("{0} {1} {2} {3} {4}\n".format(0, x_centre, y_centre, w, h))

    # save all affine transformations to disk too
    if not args.no_create_files:
        for frame in affine_transforms:
            frame_path = os.path.join(file_name, frame[0], file_extension)
            if args.verbose:
                print("Saving file {}".format(frame_path))
            cv2.imwrite(frame_path, frame[1]['image'])
            annotations_path = os.path.join(file_name, frame[0], '.txt')
            with open(annotations_path, 'a') as f:
                for bbox in frame[1]['bboxes']:
                    x_centre, y_centre, w, h = bbox
                    f.write("{0} {1} {2} {3} {4}\n".format(0, x_centre, y_centre, w, h))

    # # update progress bar
    # if not args.verbose:
    #     bar.update(1)

            

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def bbox_to_yolo(xtl, ytl, xbr, ybr):
    x_centre = (float(xtl) + float(xbr)) / 2
    y_centre = (float(ytl) + float(ybr)) / 2
    w = (float(xbr) - float(xtl))
    h = (float(ybr) - float(ytl))
    return x_centre, y_centre, w, h

if __name__ == '__main__':
    main()

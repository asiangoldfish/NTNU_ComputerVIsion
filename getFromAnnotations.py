import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# os.environ["QT_QPA_PLATFORM"] = "xcb"

def parse_annotations(xml_file, skip_interval = 1):
    """
    Parse annotations from an XML file, skipping annotations at a given interval for efficiency.

    Args:
        xml_file (str): The path to the XML file containing annotations.
        skip_interval (int, optional): The interval at which annotations should be skipped. Defaults to 1 (no skipping).

    Returns:
        tuple: A tuple containing annotation information in the format (source, frame_nr, xtl, ytl, xbr, ybr).
    """

    print("Parsing annotations from {}".format(xml_file))
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = ()    # A new `annotations` tuple for the current xml file
    source = root.find('meta').find('source').text
    print("Source: {}".format(source))

    for track in root.findall('track'):
        for idx, box in enumerate(track):
            if idx % skip_interval != 0:    # if the annotation doesn't match the given interval
                continue                    # skip the frame
            frame_nr = int(box.get('frame'))
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            annotations = annotations + ((source, frame_nr, xtl, ytl, xbr, ybr),)
    if len(annotations) == 0:
        print("Error: No annotations found in {}".format(xml_file))
    return annotations

def crop_frames(xml_path, frames_dir, output_dir):
    """
    Crop frames based on the annotations and save them to the output directory.
    Also apply various transformations for the purpose of data augmentation.

    Args:
        xml_path (str): The path to the XML file containing annotations.
        frames_dir (str): The directory containing the frames.
        output_dir (str): The directory to save the cropped frames.
    """

    annotations = parse_annotations(xml_path, 30)   # 2nd arg defines how many frames to skip
    if len(annotations) == 0:
        return

    # `idx` is needed to differentiate the names of multiple bounding boxes in the same frame
    for idx, (source, frame_nr, xtl, ytl, xbr, ybr) in enumerate(annotations, start=1):
        frame_path = os.path.join(frames_dir, 'frame_{:06d}.PNG'.format(frame_nr))
        print("Cropping frame {} from {}".format(frame_nr, frame_path))
        frame = cv2.imread(frame_path)
        cropped_frame = frame[ytl:ybr, xtl:xbr]

        # Compose output path for the new cropped image
        output_path = os.path.join(output_dir, '{}_frame_{:06d}_{}.PNG'.format(source, frame_nr, idx))

        # Data augmentation: create transformed versions of the same image
        augment(cropped_frame, output_path)

        # Show the current image on screen
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        cv2.imshow('Cropped Frame', cropped_frame)
        cv2.waitKey(1)

        # Save the new cropped image in the specified path
        cv2.imwrite(output_path, cropped_frame)


    cv2.destroyAllWindows()

def augment(img, output_path):
    """
    Produce various transformed versions of an image for data augmentation.

    Args:
        img (numpy.ndarray): The original image (already cropped to its bounding box) before any other transformations.
        output_path (str): The path where `img` is saved, needed to add details about the transformations to the new files' names.
    """

    # Split path to add details about the transformations in the file name
    file_name, file_extension = os.path.splitext(output_path)

    # Rotation
    rows, cols = img.shape[:2]
    rotation_angle = 30     # 30° for no particular reason
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    cv2.imwrite(file_name + "_rotated" + file_extension, rotated_img)

    # Scaling down
    scale_factor = 0.5
    scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    cv2.imwrite(file_name + "_scaled_down" + file_extension, scaled_img)

    # Shearing
    shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])   # Shearing along both axis
    sheared_img = cv2.warpAffine(img, shear_matrix, (cols, rows))
    cv2.imwrite(file_name + "_sheared" + file_extension, sheared_img)

    # Brightness & contrast adjustments
    brightness = 50     # Increase brightness by 50 units (?)
    contrast = 1.5      # Increase contrast by a factor of 1.5
    adjusted_img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    cv2.imwrite(file_name + "_adjusted" + file_extension, adjusted_img)

    # Gaussian noise
    noise_img = np.zeros(img.shape, dtype=np.uint8)
    cv2.randn(noise_img, 0, 25)     # Generate Gaussian noise with mean 0 and standard deviation 25
    noisy_img = cv2.add(img, noise_img)
    cv2.imwrite(file_name + "_noisy" + file_extension, noisy_img)

    # Horisontally flipped
    flipped_h_img = cv2.flip(img, 1)
    cv2.imwrite(file_name + "_flipped_h" + file_extension, flipped_h_img)

    # Vertically flipped
    flipped_v_img = cv2.flip(img, 0)
    cv2.imwrite(file_name + "flipped_v" + file_extension, flipped_v_img)

    # Horisontally & vertically flipped
    flipped_h_v_img = cv2.flip(flipped_h_img, 0)
    cv2.imwrite(file_name + "flipped_h_v" + file_extension, flipped_h_v_img)

    # Add a blue tint typical of underwater images
    # blue_tint = np.array([255, 128, 0])
    blue_tint_img = np.copy(img)
    # blue_tint_img = cv2.addWeighted(img, 1, np.zeros_like(img), 0, 0, blue_tint)
    blue_tint_img[:, :, 0] = cv2.add(blue_tint_img[:, :, 0], 100)
    blue_tint_img[:, :, 1] = cv2.add(blue_tint_img[:, :, 1], 25)
    blue_tint_img[:, :, 2] = cv2.add(blue_tint_img[:, :, 2], 25)
    cv2.imwrite(file_name + "_blue_tint" + file_extension, blue_tint_img)

    # Add a algae-green tint typical of underwater images in the sea
    # green_tint = np.array([0, 255, 128])
    green_tint_img = np.copy(img)
    # green_tint_img = cv2.addWeighted(img, 1, np.zeros_like(img), 0, 0, green_tint)
    green_tint_img[:, :, 1] = cv2.add(green_tint_img[:, :, 1], 50)
    green_tint_img[:, :, 0] = cv2.add(green_tint_img[:, :, 0], 25)
    cv2.imwrite(file_name + "_green_tint" + file_extension, green_tint_img)
    

def process_video_annotations(annotations_dir, output_dir):
    """
    Process annotations for all videos in a directory.

    Args:
        annotations_dir (str): The directory containing video annotations.
        output_dir (str): The directory to save the cropped frames.
    """

    os.makedirs(output_dir, exist_ok=True)

    for video in os.listdir(annotations_dir):
        if not os.path.isdir(os.path.join(annotations_dir, video)):
            continue
        annotations_path = os.path.join(annotations_dir, video, 'annotations.xml')
        frames_dir = os.path.join(annotations_dir, video, 'images')
        crop_frames(annotations_path, frames_dir, output_dir)

if __name__ == '__main__':
    annotations_dir = '../ANADROM/Annotation'
    output_dir = 'CroppedFrames'

    process_video_annotations(annotations_dir, output_dir)

# TODO
# - Comment & doxygen
# - Explore data augmentation and preprocessing to extract features
# - Find software to define bounding boxes on other objects than fish

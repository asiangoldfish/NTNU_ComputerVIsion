import os
import cv2
import xml.etree.ElementTree as ET

# os.environ["QT_QPA_PLATFORM"] = "xcb"

def parse_annotations(xml_file):
    print("Parsing annotations from {}".format(xml_file))
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = ()
    source = root.find('meta').find('source').text
    print("Source: {}".format(source))

    for track in root.findall('track'):
        for box in track:
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
    annotations = parse_annotations(xml_path)
    if len(annotations) == 0:
        return
    for idx, (source, frame_nr, xtl, ytl, xbr, ybr) in enumerate(annotations, start=1):
        frame_path = os.path.join(frames_dir, 'frame_{:06d}.PNG'.format(frame_nr))
        print("Cropping frame {} from {}".format(frame_nr, frame_path))
        frame = cv2.imread(frame_path)
        cropped_frame = frame[ytl:ybr, xtl:xbr]
        output_path = os.path.join(output_dir, '{}_frame_{:06d}_{}.PNG'.format(source, frame_nr, idx))
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        cv2.imshow('Cropped Frame', cropped_frame)
        cv2.waitKey(1)
        cv2.imwrite(output_path, cropped_frame)
    cv2.destroyAllWindows()

def process_video_annotations(annotations_dir, output_dir):
    # output_dir = os.path.join(output_dir, 'cropped_frames')
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

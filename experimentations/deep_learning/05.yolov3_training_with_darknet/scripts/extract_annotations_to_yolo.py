import xml.etree.ElementTree as ET
import os

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Parse the XML file
tree = ET.parse('annotations.xml')
root = tree.getroot()

# Find all track elements with label="SALMON"
salmon_tracks = root.findall(".//track")

directory = 'annotations'
os.makedirs(directory, exist_ok=True)

dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"Current working directory: {dir_path}")

# Iterate over each track with label="SALMON"
for track in salmon_tracks:
    track_id = track.get('id')
    # Iterate over each box in the track
    for box in track.findall('box'):
        frame = box.get('frame')
        # Extract bounding box coordinates
        # Format: <object-class> <x_center> <y_center> <width> <height>


        xtl = box.get('xtl')
        ytl = box.get('ytl')
        xbr = box.get('xbr')
        ybr = box.get('ybr')

        # Convert to yolo format and normalize
        x_center = (float(xtl) + float(xbr)) / 2 / IMAGE_WIDTH
        y_center = (float(ytl) + float(ybr)) / 2 / IMAGE_HEIGHT
        width = (float(xbr) - float(xtl)) / IMAGE_WIDTH
        height = (float(ybr) - float(ytl)) / IMAGE_HEIGHT

        # Write bounding box coordinates to a text file
        # Frame number occupies 6 digits
        frame_num = str(frame).zfill(6)
        with open(f'annotations/frame_{frame_num}.txt', 'a') as f:
            f.write(f'0 {x_center} {y_center} {width} {height}\n')

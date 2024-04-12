import xml.etree.ElementTree as ET
import os

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

subdirs = list()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    type=str,
    default="",
    help="The location where we want to store the new annotations",
    required=True
)
parser.add_argument(
    "--source",
    type=str,
    default="",
    help="The location where we want to store the new annotations",
    required=True
)

args = parser.parse_args()

# The location where we want to store the new annotations
TARGET = args.target
ANNOTATED_VIDEOS = args.source

for a in os.listdir(ANNOTATED_VIDEOS):
    subdirs.append(a)

if not os.path.exists(TARGET):
    os.mkdir(TARGET)

for i, subdir in enumerate(subdirs):
    # Logging
    print(f"Video {i + 1}/{len(subdirs)}")

    # Parse the XML file
    tree = ET.parse(os.path.join(ANNOTATED_VIDEOS, subdir, "annotations.xml"))
    root = tree.getroot()

    source, _ = os.path.splitext(root.find('meta').find('source').text)

    # Check if the source exists in 'dataset'
    videos = os.listdir('dataset')
    if f"{source}.mp4" not in videos:
        continue

    # Find all track elements with label="SALMON"
    salmon_tracks = root.findall(".//track")

    # Iterate over each track with label="SALMON"
    for track in salmon_tracks:
        track_id = track.get("id")
        # Iterate over each box in the track
        for box in track.findall("box"):
            frame = box.get("frame")
            # Extract bounding box coordinates
            # Format: <object-class> <x_center> <y_center> <width> <height>

            xtl = box.get("xtl")
            ytl = box.get("ytl")
            xbr = box.get("xbr")
            ybr = box.get("ybr")

            # Convert to yolo format and normalize
            x_center = (float(xtl) + float(xbr)) / 2 / IMAGE_WIDTH
            y_center = (float(ytl) + float(ybr)) / 2 / IMAGE_HEIGHT
            width = (float(xbr) - float(xtl)) / IMAGE_WIDTH
            height = (float(ybr) - float(ytl)) / IMAGE_HEIGHT


            # Write bounding box coordinates to a text file
            # Frame number occupies 6 digits
            frame_num = str(frame).zfill(6)
            with open(f"output/{source}_noiseIncreased_frame{frame_num}.txt", "a") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

            with open(f"output/{source}_noiseReduced_frame{frame_num}.txt", "a") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

            with open(f"output/{source}_illuminationIncreased_frame{frame_num}.txt", "a") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

            with open(f"output/{source}_illuminationReduced_frame{frame_num}.txt", "a") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

            with open(f"output/{source}_contrasted_frame{frame_num}.txt", "a") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

            with open(f"output/{source}_frame{frame_num}.txt", "a") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")


print("")
#!/bin/bash

# Specify the destination directory
destination_dir="annotations"

# Loop through each directory containing annotations.xml files
i=0
for dir in ~/mounted_drives/documents/computer_vision/fish_classification_data/annotated_videos/task_fish_detection-2024_0*/; do
    # Get the index for the destination filename
    # index=$(basename "$dir" | sed 's/task_fish_detection-2024_//')
    
    # Copy the annotations.xml file to the destination directory with an incremental name
    cp "$dir"/annotations.xml "annotations_$i.xml"

    ((i++))
done

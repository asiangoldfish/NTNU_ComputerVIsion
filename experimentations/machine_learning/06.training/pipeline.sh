#!/usr/bin/env bash

# Extract annotations
echo "Extracting annotatations and converting to YOLO format"
python extract_annotations_to_yolo.py --target output --source /home/khai/mounted_drives/documents/computer_vision/fish_classification_data/annotated_videos || exit 1
echo ""

# Augment data
echo "Augmenting data..."
python augment_videos.py || exit 1

# Copy original files to output

echo "Processing training dataset completed!"
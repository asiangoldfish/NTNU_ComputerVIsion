#!/usr/bin/env bash

yolo detect train data=yolo.yaml model=yolov8n.pt epochs=10 imgsz=640
#!/usr/bin/env bash

if [ -z "$1" ]
    then
        echo "No argument supplied"
        exit 1
fi

if [ -z "$2" ]
    then
        echo "No argument supplied"
        exit 1
fi

yolo predict model="$2" source="backup/$1" imgsz=1080 # conf=0.2
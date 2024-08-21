#!/bin/bash

# Set the source and destination directories
SRC_DIR="$HOME/coloradar/bags"
DST_DIR="$HOME/coloradar/bags2"

# Ensure the destination directory exists
mkdir -p "$DST_DIR"

# Iterate over all .bag files in the source directory
for bagfile in "$SRC_DIR"/*.bag; do
    # Get the base filename without the extension
    base_filename=$(basename "$bagfile" .bag)

    # Convert the ROS1 bag file to a ROS2 bag and save it in the destination directory with the .db3 extension
    rosbags-convert "$bagfile" --dst "$DST_DIR/$base_filename.db3"

    # Print a message indicating the conversion is complete for this file
    echo "Converted $bagfile to $DST_DIR/$base_filename.db3"
done

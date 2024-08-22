#!/bin/bash

# Get the absolute path of the directory containing the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the source and destination directories
BASE_DIR="$HOME/coloradar"
SRC_DIR="$BASE_DIR/bags"
DST_DIR="$BASE_DIR/bags2"
TYPE_FIX_SCRIPT="$SCRIPT_DIR/fix_custom_message_types.py"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Dataset directory $BASE_DIR does not exist."
    exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Bag files at $SRC_DIR not found."
    exit 1
fi
if [ ! -f "$TYPE_FIX_SCRIPT" ]; then
    echo "Error: Python script $TYPE_FIX_SCRIPT not found."
    exit 1
fi
mkdir -p "$DST_DIR"

# Iterate over all .bag files in the source directory
for bagfile in "$SRC_DIR"/*.bag; do
    # Get the base filename without the extension
    base_filename=$(basename "$bagfile" .bag)

    # Convert the ROS1 bag file to a ROS2 bag and save it in the destination directory without an extension
    rosbags-convert "$bagfile" --dst "$DST_DIR/$base_filename"

    # Print a message indicating the conversion is complete for this file
    echo "Converted $bagfile to $DST_DIR/$base_filename"

    # Fix metadata.yaml using the absolute path to the Python script
    python "$TYPE_FIX_SCRIPT" "$DST_DIR/$base_filename" "dca1000_device" "mapless_navigation"
done

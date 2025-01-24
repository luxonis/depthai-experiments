#!/bin/bash

# Define the oakapp file
OAKAPP_TEMP=oakapp_template.toml
OAKAPP=oakapp.toml

# Ensure the oakapp file exists
if [ ! -f "$OAKAPP_TEMP" ]; then
    echo "File not found: $OAKAPP_TEMP"
    exit 1
fi

# Create a temporary oakapp file
cp $OAKAPP_TEMP $OAKAPP

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --model)
        MODEL="$2"
        shift
        ;;
    --device)
        DEVICE="$2"
        shift
        ;;
    --annotation_mode)
        ANNOTATION_MODE="$2"
        shift
        ;;
    --fps_limit)
        FPS_LIMIT="$2"
        shift
        ;;
    --media)
        MEDIA="$2"
        shift
        ;;
    --api_key)
        API_KEY="$2"
        shift
        ;;
    esac
    shift
done

# Use sed to replace placeholders with the provided values
if [ -n "$MODEL" ]; then
    sed -i "s|<Model>|--model $MODEL|g" "$OAKAPP"
else
    echo "Error: --model argument is required."
    rm $OAKAPP
    exit 1
fi

if [ -n "$ANNOTATION_MODE" ]; then
    sed -i "s|<Mode>|--annotation_mode $ANNOTATION_MODE|g" "$OAKAPP"
else
    sed -i "s|<Mode>||g" "$OAKAPP"
fi

if [ -n "$FPS_LIMIT" ]; then
    sed -i "s|<FPS>|--fps_limit $FPS_LIMIT|g" "$OAKAPP"
else
    sed -i "s|<FPS>||g" "$OAKAPP"
fi

if [ -n "$MEDIA" ]; then
    sed -i "s|<Media>|--media /app/$MEDIA|g" "$OAKAPP"
else
    sed -i "s|<Media>||g" "$OAKAPP"
fi

if [ -n "$API_KEY" ]; then
    sed -i "s|<API_KEY>|--api_key /app/$API_KEY|g" "$OAKAPP"
else
    sed -i "s|<API_KEY>||g" "$OAKAPP"
fi

# Connect to the device
if [ -n "$DEVICE" ]; then
    oakctl connect $DEVICE
else
    oakctl connect
fi

# Run the example
oakctl app run .

# Remove the temporary oakapp file
rm $OAKAPP

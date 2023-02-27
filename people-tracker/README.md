# People tracker

This application counts how many people went up / down / left / right in the video stream, allowing you to
receive an information about eg. how many people went into a room or went through a corridor.

Demo uses [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the Open Model Zoo for person detection.

## Demo

![demo](https://user-images.githubusercontent.com/18037362/199674225-ebb1b811-6e45-4535-abe3-f18d1644634d.gif)

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```
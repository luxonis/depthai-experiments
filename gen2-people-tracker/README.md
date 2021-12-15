# Gen2 People tracker

This application counts how many people went up / down / left / right in the video stream, allowing you to
receive an information about how many people went into a room or went through a corridor.

Demo uses [Script](https://docs.luxonis.com/projects/api/en/latest/components/nodes/script/) node to "decode" movement from trackelts information. In script node, when a new tracklet is added, it's coordiantes are saved. When it's removed/lost, it compares starting coordinates
with end coordinates and if movement was greater than `THRESH_DIST_DELTA`, it means a person movement was valid and added to counter.

This demo can also send counter results through the SPI.

Demo uses [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the Open Model Zoo for person detection.

## Demo

![demo](https://user-images.githubusercontent.com/18037362/145656510-94e12444-7524-47f9-a036-7ed8ee78fd7a.gif)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```

# Gen2 People tracker

This application counts how many people went upwards / downwards / leftwards / rightwards in the video stream, allowing you to
receive an information about how many people went into a room or went through a corridor.

This demo can also send tracklets results through the SPI.

The model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the OpenVINO Model Zoo.

## Demo

[![Watch the demo](https://user-images.githubusercontent.com/18037362/116413235-56e96e00-a82f-11eb-8007-bfcdb27d015c.gif)](https://www.youtube.com/watch?v=MHmzp--pqUA)

## Pre-requisites

Purchase a DepthAI model (see https://shop.luxonis.com/)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```

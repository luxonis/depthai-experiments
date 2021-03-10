# People tracker

This application counts how many people went upwards / downwards / leftwards / rightwards in the video stream, allowing you to
receive an information about how many people went into a room or went through a corridor.

The model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) from the OpenVINO Model Zoo.

## Demo

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90752628-ee2d1780-e2d7-11ea-8e48-ca94b02a7674.gif)](https://youtu.be/8RiHkkGKdj0)

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

By default, you should see a debug window and console output that shows how many people were
tracked. Also, you should see `results.json` file with timestamped results.

If you want to run it without preview, just to collect the data, you can modify `main.py` and set

```diff
- debug = True
+ debug = False
```

Then the app will run without preview window nor debug messages, will just save the results to `results.json`

## Credits

Adrian Rosebrock, OpenCV People Counter, PyImageSearch, https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/, accessed on 6 August 2020

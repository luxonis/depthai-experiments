# People tracker

This application counts how many people went upwards / downwards on the image, allowing you to
receive an information about how many people went into a room or went through a corridor.

It utilizes DepthAI `object_detector` stream so all the tracking is done on the DepthAI side

Model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

## Pre-requisites

1. Purchase a DepthAI model (see https://shop.luxonis.com/)
2. Install DepthAI API (see [here](https://docs.luxonis.com/api/) for your operating system/platform)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```

By default, you should see a debug window that shows how many people were
detected and classsified as moving `up` or `down`.

If you want to run it without preview, just to collect the data, you can modify `main.py` and set

```diff
- debug = True
+ debug = False
```

Then the app will run without preview window, printing only message to the terminal.
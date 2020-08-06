# People tracker

This application counts how many people went upwards / downwards / leftwards / rightwards on the image, allowing you to
receive an information about how many people went into a room or went through a corridor.

Model used in this example is [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

## Demo

[![Watch the demo](https://img.youtube.com/vi/8RiHkkGKdj0/hqdefault.jpg)](https://youtu.be/8RiHkkGKdj0)

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

By default, you should see a debug window and console output that shows how many people were
tracked. Also, you should see `results.json` file with timestamped results.

If you want to run it without preview, just to collect the data, you can modify `main.py` and set

```diff
- debug = True
+ debug = False
```

Then the app will run without preview window nor debug messages, will just save the results to `results.json`
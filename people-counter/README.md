# People counting

One of the basic examples of DepthAI usage - counting people in front of the camera and save the
results to JSON file (for further processing)

App can be useful as a starting point to other applications or to monitor e.x. the conference
rooms usage

Models used in this example are:
- [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
- [pedestrian_detection_adas_0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_pedestrian_detection_adas_0002_description_pedestrian_detection_adas_0002.html)
- [mobilenet_ssd](https://docs.openvinotoolkit.org/latest/omz_models_public_mobilenet_ssd_mobilenet_ssd.html)

## Demo

[![Watch the demo](https://user-images.githubusercontent.com/5244214/90751105-fc7a3400-e2d5-11ea-82fe-3c7797e99e3e.gif)](https://youtu.be/M0xQI1kXju4)

## Pre-requisites

Purchase a DepthAI model (see https://shop.luxonis.com/)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

With default network
```
python3 main.py
```

With specific network (can be either `person_detection_retail_0013`, `pedestrian_detection_adas_0002` or `mobilenet_ssd`)
```
python3 main.py -m mobilenet_ssd
```

You should see a debug window and console output that shows how many people were
detected. Also, you should see `results.json` file with timestamped results.

If you want to run it without preview, just to collect the data, you can modify `main.py` and set

```diff
- debug = True
+ debug = False
```

Then the app will run without preview window nor debug messages, will just save the results to `results.json`

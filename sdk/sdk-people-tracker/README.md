# People tracker

This application counts how many people went up/down/left/right in the video stream, allowing you to
receive information about how many people went into a room or went through a corridor.

Demo uses DepthAI SDK
and [person_detection_retail_0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
from the Open Model Zoo for person detection.

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

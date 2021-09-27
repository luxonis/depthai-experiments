[中文文档](README.zh-CN.md)

# Pedestrian reidentification

This example demonstrates how to run 2 stage inference on DepthAI using Gen2 Pipeline Builder.

Original OpenVINO demo, on which this example was made, is [here](https://docs.openvinotoolkit.org/2020.1/_demos_pedestrian_tracker_demo_README.html).

## Demo

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Person Re-ID on DepthAI")

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
main.py [-h] [-nd] [-cam] [-vid VIDEO] [-w WIDTH] [-lq]
```

Optional arguments:
 - `-h, --help` Show this help message and exit
 - `-nd, --no-debug` Prevent debug output
 - `-cam, --camera` Use DepthAI RGB camera for inference (conflicts with -vid)
 - `-vid VIDEO, --video VIDEO` Path to video file to be used for inference (conflicts with -cam)
 - `-w WIDTH, --width WIDTH` Visualization width
 - `-lq, --lowquality` Uses resized frames instead of source


To use with a video file, run the script with the following arguments

```
python3 main.py -vid input.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 

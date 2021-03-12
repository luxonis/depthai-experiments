# Vehicle License Plate identification

This example demonstrates how to run a N stage inference on DepthAI using Gen2 Pipeline Builder.

Original OpenVINO demo, on which this example was made, is [here](https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html).

This project is intended to work in the "Barrier" setting and aims to detect vehicle and license plates and save those detections. At a later point, this code may be extended to:

1/ Upload the saved detection image + data to a server

2/ Do more robust detecting



## Demo

TODO(wferrell): Update with an example video and screenshot.

[![License Plate Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Vehicle and license plate detection and identification on DepthAI")

## Pre-requisites

1. Purchase a DepthAI camera (see [shop.luxonis.com](https://shop.luxonis.com/))
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


## Other links in Researching this

https://github.com/livezingy/license-plate-recognition/blob/master/testOpenVINO.py#L78
https://answers.opencv.org/question/205754/how-to-run-pretrained-model-with-openvino-on-rpi/
https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html

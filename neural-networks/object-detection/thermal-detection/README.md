# Thermal Person Detection

This example demonstrates person detection with a thermal camera. We trained a custom YOLOv6-Nano based object detection model on a mixed dataset made from our synthetic images, [FLIR](https://www.flir.eu/oem/adas/adas-dataset-form/) dataset and some smaller datasets from [Roboflow](https://universe.roboflow.com/search?q=class%3Athermal+camera). You can find more information about the model [here](https://hub.luxonis.com/ai/models/b1d7a62f-7020-469c-8fa9-a6d1ff3499b2?view=page).

## Demo

![Exampe](media/thermal_person.gif)

## Installation

Running this example requires a **Luxonis Thermal device** connected to your computer. You can find more information about it [here](https://docs.luxonis.com/hardware/products/OAK%20Thermal).

Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 main.py
```

This will run the experiment with the default thermal person detection model.

Here is a list of all available parameters:

```
-m MODEL, --model MODEL
                      HubAI model reference. (default: luxonis/thermal-person-detection:256x192)
-api API_KEY, --api_key API_KEY
                      HubAI API key to access private model. (default: )
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                      FPS limit for the model runtime. (default: 20)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

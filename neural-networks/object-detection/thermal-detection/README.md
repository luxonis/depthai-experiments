# Thermal Person Detection

## Overview

This example demonstrates person detection with a thermal camera. We trained a custom YOLOv6-Nano based object detection model on a mixed dataset made from our synthetic images, [FLIR](https://www.flir.eu/oem/adas/adas-dataset-form/) dataset and some smaller datasets from [Roboflow](https://universe.roboflow.com/search?q=class%3Athermal+camera).

## Installation

Running this example requires a **Luxonis Thermal device** connected to your computer. You can find more information about it [here](https://docs.luxonis.com/hardware/products/OAK%20Thermal).

Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

# Usage

The example can be ran with:

```bash
python3 main.py --model <MODEL> --device <DEVICE>
```

Here are all the arguments:

- `<Model>`: A unique HubAI identifier of the model;
- `<API>` \[OPTIONAL\]: HubAI API key to access private model;
- `<Device>` \[OPTIONAL\]: DeviceID or IP of the camera to connect to.
  By default, the first locally available device is used;
- `<FPS>` \[OPTIONAL\]: The upper limit for camera captures in frames per second (FPS).
  The limit is not used when infering on media.
  By default, the FPS is not limited.
- `<Media>` \[OPTIONAL\]: Path to the media file to be used as input.
  Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
  By default, camera input is used;

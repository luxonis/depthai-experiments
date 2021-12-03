## [Gen2] Yolo object detection

This repository contains code for various Yolo experiments. For better understanding, please look in table below:

| Directory       | Decoding | Versions                     | Description                                                  |
| --------------- | -------- | ---------------------------- | ------------------------------------------------------------ |
| `car-detection`   | device   | V3-tiny, V4-tiny             | Car detection using YoloV3-tiny and YoloV4-tiny.             |
| `host-decoding`   | host     | V5                           | Object detection using YoloV5 and on-host decoding.          |
| `device-decoding` | device   | V3, V3-tiny, V4, V4-tiny, V5 | General object detection using any of the version for which we support on-device decoding. |

DepthAI allows execution of certain Yolo object detection models fully on device, including decoding. Currently, the supported models are:

* YoloV3 & YoloV3-tiny,

* YoloV4 & YoloV4-tiny,
* YoloV5.

Non-supported Yolo models usually require on-host decoding.

## Pre-requisites

1. Purchase a DepthAI (or OAK) model (see [shop.luxonis.com](https://shop.luxonis.com/)).

2. Install requirements

   ```python
   python3 -m pip install -r requirements.txt
   ```
3. Open the directory and follow the instructions.
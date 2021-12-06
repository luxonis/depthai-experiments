## Yolo object detection

This repository contains code for various Yolo experiments:

| Directory       | Decoding | Version                     | Description                                                  |
| :-------------: | :------: | :--------------------------: | ------------------------------------------------------------ |
| `car-detection`   | device   | V3-tiny, V4-tiny             | Car detection using YoloV3-tiny and YoloV4-tiny with on-device decoding. |
| `device-decoding` | device   | V3, V3-tiny, V4, V4-tiny, V5 | General object detection using any of the versions for which we support on-device decoding. **Use this if you want to see an example of how to run your own model!** |
| `host-decoding`   | host     | V5                           | Object detection using YoloV5 and on-host decoding.          |
| `yolox` | host | X | Object detection without anchors using YOLOX-tiny with on-host decoding. |
| `yolop` | host | P | Vehicle detection, road segmentation, and lane segmentation using YOLOP on OAK with on-host decoding. |

## On-Device decoding

DepthAI allows execution of certain Yolo object detection models fully on a device, including decoding. Currently, the supported models are:

* YoloV3 & YoloV3-tiny,

* YoloV4 & YoloV4-tiny,
* YoloV5.

Non-supported Yolo models usually require on-host decoding. We provide on-device decoding pipeline examples in `device-decoding` (and similar code is used in `car-detection`). Other repositories are likely to use on-host decoding.

## Pre-requisites

1. Purchase a DepthAI (or OAK) device (see [shop.luxonis.com](https://shop.luxonis.com/)).
2. Open the directory and follow the instructions.


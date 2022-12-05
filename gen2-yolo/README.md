![yolo-logo](https://user-images.githubusercontent.com/56075061/144863247-fa819d1d-28d6-498a-89a8-c3f94d9e9357.gif)

## Yolo Experiments

This repository contains code for various Yolo experiments:

| Directory       | Decoding | Version                     | Description                                                  |
| :-------------: | :------: | :--------------------------: | ------------------------------------------------------------ |
| `main.py` | device   | From https://tools.luxonis.com | Run your custom trained YOLO model that was converted using the tools.luxonis.com. Uses [DepthAI-SDK](https://docs.luxonis.com/projects/sdk/en/latest/) |
| `device-decoding` | device   | V3, V3-tiny, V4, V4-tiny, V5 | General object detection using any of the versions for which we support on-device decoding. Uses [DepthAI-API]https://docs.luxonis.com/projects/api/en/latest/) |
| `car-detection`   | device   | V3-tiny, V4-tiny             | Car detection using YoloV3-tiny and YoloV4-tiny with on-device decoding ([DepthAI-SDK](https://docs.luxonis.com/projects/sdk/en/latest/)). |
| `host-decoding`   | host     | V5                           | Object detection using YoloV5 and on-host decoding.          |
| `yolox` | host | X | Object detection without anchors using YOLOX-tiny with on-host decoding. |
| `yolop` | host | P | Vehicle detection, road segmentation, and lane segmentation using YOLOP on OAK with on-host decoding. |

## On-Device decoding

DepthAI allows execution of certain Yolo object detection models fully on a device, including decoding. Currently, the supported models are:

* YoloV3 & YoloV3-tiny,

* YoloV4 & YoloV4-tiny,
* YoloV5.

Non-supported Yolo models usually require on-host decoding. We provide on-device decoding pipeline examples in `device-decoding` (and similar code is used in `car-detection`). Other repositories are likely to use on-host decoding.

## Depth information

DepthAI enables you to take the advantage of depth information and get `x`, `y`, and `z` coordinates of detected objects. Experiments in this directory are not using the depth information. If you are interested in using the depth information with Yolo detectors, please check our [documentation](https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_tiny_yolo/#rgb-tinyyolo-with-spatial-data).

![SpatialObjectDetection](https://user-images.githubusercontent.com/56075061/144864639-4519699e-d3da-4172-b66b-0495ea11317e.png)

## Usage

Open the directory and follow the instructions.


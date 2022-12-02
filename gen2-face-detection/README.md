# YuNet Face detection on DepthAI

This example shows an implementation of [libfacedetection](https://github.com/ShiqiYu/libfacedetection) on DepthAI API.

Current input to neural network is 160 x 120 (W x H). Camera preview is set to 640 x 480 (W x H) and is resized to neural network input shape with ImageManip.

![Image example](imgs/example.gif)

Image is taken from [here](https://www.pexels.com/photo/multi-cultural-people-3184419/).

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py [options]
```

Options:

* `-conf, --confidence_thresh`: Set the confidence threshold. Default: *0.6*.
* `-iou, --iou_thresh`: Set the NMS IoU threshold. Default: *0.3*.
* `-topk, --keep_top_k`: Keep at most top_k picked indices. Default: *750*.

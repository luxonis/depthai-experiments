## YOLOP on DepthAI

This example shows an implementation of [YOLOP](https://github.com/hustvl/YOLOP) on DepthAI in the Gen2 API system. ONNX models is taken from the official repository and converted to blob so it can run on OAK devices.

The model performs:

* lane segmentation,
* line segmentation, and
* vehicle detection.

Input shape of the model is 320 x 320, and we resize the input video to the required shape. [Letterboxing](https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/#letterboxing) might be useful for slightly better performance.

![Image example](https://user-images.githubusercontent.com/56075061/144842281-3413133e-7a44-4030-a572-9887fad1bbc6.gif)

Example shows input video with overlay of lane and line segmentation and vehicle detections. Example video is taken from [YOLOP repository](https://github.com/hustvl/YOLOP/tree/main/inference/videos).

## Pre-requisites

1. Install requirements:
   ```
   python3 -m pip install -r requirements.txt
   ```
2. Download sample videos
   ```
   python3 download.py
   ```

## Usage

```
python3 main.py [options]
```

Options:

* -v, --video_path: Path to the video input for inference. Default: *vids/1.mp4*.
* -conf, --confidence_thresh: Set the confidence threshold. Default: 0.5.
* -iou, --iou_thresh: Set the NMS IoU threshold. Default: 0.3.

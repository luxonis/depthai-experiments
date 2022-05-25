[中文文档](README.zh-CN.md)

# Pedestrian reidentification

This example demonstrates how to run 2 stage inference on DepthAI. It runs [person-detection-retail-0013 model](https://docs.openvino.ai/latest/omz_models_model_person_detection_retail_0013.html) and [person-reidentification-retail-0288](https://docs.openvino.ai/latest/omz_models_model_person_reidentification_retail_0288.html) on the device itself.

Original OpenVINO demo, on which this example was made, is [here](https://docs.openvinotoolkit.org/2020.1/_demos_pedestrian_tracker_demo_README.html).

## Demo

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/18037362/170273561-71bb3b9a-0b40-4221-8a5a-9a543fc4fb4a.gif)](https://youtu.be/Ql9LQtao8-s)


### How it works

1. Color camera produces high-res frames, sends them to host, Script node and downscale ImageManip node
2. Downscale ImageManip will downscale from high-res frame to 544x320, required by 1st NN in this pipeline; object detection model [person-detection-retail-0013 model](https://docs.openvino.ai/latest/omz_models_model_person_detection_retail_0013.html)
3. 544x320 frames are sent from downscale ImageManip node to the object detection model (MobileNetSpatialDetectionNetwork)
4. Object detections are sent to the Script node
5. Script node first syncs object detections msg with frame. It then goes through all detections and creates ImageManipConfig for each detected face. These configs then get sent to ImageManip together with synced high-res frame
6. ImageManip will crop only the face out of the original frame. It will also resize the face frame to required size (128,256) by the person-reidentification NN model
7. Face frames get send to the 2nd NN - [person-reidentification](https://docs.openvino.ai/latest/omz_models_model_person_reidentification_retail_0288.html) NN model. NN results are sent back to the host
8. Frames, object detections, and reidentification results are all **synced on the host** side and then displayed to the user

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python main.py
```

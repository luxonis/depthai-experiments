[中文文档](README.zh-CN.md)

# Pedestrian reidentification

This demo first runs person detection model, then runs reidentification model on the cropped person frame, which provides
256 FP16 vector. This vector is then compared with other (people) vectors to re-identify the person. Comparison is done on the device as well, using custom cosinus similarity model. All these processes are entirely run on the device itself (2 Script nodes are used for business logic), and only results and frames are sent back to the host.

There is a ~1 second delay between frames are bounding boxes, that's because we run person reidentification followed by multiple cosinus similarity processes on the device, before sending bounding box of reidentified person back to the host.

Original OpenVINO demo, on which this example was made, is [here](https://docs.openvinotoolkit.org/2020.1/_demos_pedestrian_tracker_demo_README.html).
Models used:
- [person_detection_retail_0013](https://docs.openvino.ai/latest/omz_models_model_person_detection_retail_0013.html)
- [person_reidentification_retail_0031](https://docs.openvino.ai/2020.1/_models_intel_person_reidentification_retail_0031_description_person_reidentification_retail_0031.html)
- Custom [cosinus similarity](https://github.com/luxonis/depthai-experiments/tree/master/gen2-custom-models/generate_model/pytorch_cos_dist.py) model implemented with PyTorch

## Demo

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Person Re-ID on DepthAI")

## Pre-requisites

Install requirements
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```

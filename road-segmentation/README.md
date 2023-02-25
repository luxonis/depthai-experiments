##  Road Segmentation on DepthAI

This example shows how to run `road-segmentation-adas-0001` on DepthAI in the Gen2 API.

![Road Segmentation on DepthAI](https://user-images.githubusercontent.com/5244214/130064359-b9534b08-0783-4c86-979b-08cbcaff9341.gif)

## Pre-requisites

Install requirements
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```

This will download the road segmentation model from OpenVINO Model Zoo and perform the inference on RGB camera, together
with depth output
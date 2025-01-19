# Lossless Zooming

This demo shows how you can achieve lossless zooming on the device. Demo will zoom into the first face it detects. It will crop 4K frames into 1080P, centered around the face. Demo uses [face-detection-retail-0004](https://docs.openvino.ai/latest/omz_models_model_face_detection_retail_0004.html) NN model.

## Demo

[![Lossless Zooming](https://user-images.githubusercontent.com/18037362/144095838-d082040a-9716-4f8e-90e5-15bcb23115f9.gif)](https://youtu.be/8X0IcnkeIf8)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

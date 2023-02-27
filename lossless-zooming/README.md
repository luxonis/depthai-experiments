# Lossless Zooming

This demo shows how you can achieve lossless zooming on the device. Demo will zoom into the first face it detects. It
will crop 4K frames into 1080P, centered around the face. Demo
uses [face-detection-retail-0004](https://docs.openvino.ai/latest/omz_models_model_face_detection_retail_0004.html) NN
model.

## Demo

[![Lossless Zooming](https://user-images.githubusercontent.com/18037362/144095838-d082040a-9716-4f8e-90e5-15bcb23115f9.gif)](https://youtu.be/8X0IcnkeIf8)

### MJPEG

You can turn `MJPEG` on or off. It's set to `True` by default, so cropped 1080P stream will get encoded into MJPEG on
the device. On the host, it will just get decoded and shown to the user, but you could also save the MJPEG stream or
stream it elsewhere.

## Usage

Choose one of the following options:

```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```
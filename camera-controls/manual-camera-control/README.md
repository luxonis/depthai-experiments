#TODO vobec to nesedi k projektu zatial

# Lossless Zooming

This demo shows how you can achieve lossless zooming on the device. Demo will zoom into the first face it detects. It will crop 4K frames into 1080P, centered around the face. Demo uses [face-detection-retail-0004](https://docs.openvino.ai/latest/omz_models_model_face_detection_retail_0004.html) NN model.

## Demo

[![Lossless Zooming](https://user-images.githubusercontent.com/18037362/144095838-d082040a-9716-4f8e-90e5-15bcb23115f9.gif)](https://youtu.be/8X0IcnkeIf8)

### MJPEG

You can turn `MJPEG` on or off. It's set to `True` by default, so cropped 1080P stream will get encoded into MJPEG on the device. On the host, it will just get decoded and shown to the user, but you could also save the MJPEG stream or stream it elsewhere.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

```
User controls
'c' - to capture a set of images (from isp and/or raw streams)
't' - to trigger autofocus
'ioklnm,.' for manual exposure/focus:
  Control:      key[dec/inc]  min..max
  exposure time:     i   o      1..33000 [us]
  sensitivity iso:   k   l    100..1600
  focus:             ,   .      0..255 [far..near]
  white balance:     n   m   1000..12000 (light color temperature K)
To go back to auto controls:
  'e' - autoexposure
  'f' - autofocus (continuous)
Other controls:
'1' - AWB lock (true / false)
'2' - AE lock (true / false)
'3' - Select control: AWB mode
'4' - Select control: AE compensation
'5' - Select control: anti-banding/flicker mode
'6' - Select control: effect mode
'7' - Select control: brightness
'8' - Select control: contrast
'9' - Select control: saturation
'0' - Select control: sharpness
'[' - Select control: luma denoise
']' - Select control: chroma denoise

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase
```

Run the application

```
python3 main.py
```

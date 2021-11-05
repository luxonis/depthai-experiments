## [Gen2] Mobile Object Localizer on DepthAI

This example shows an implementation of Mobile Object Localizer from [Tensorflow Hub](https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1).

Blob is taken from [Pinto Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer).

Input video is resized to 192 x 192 (W x H). Output is a list of 100 scores and bounding boxes (see implementation for details).

![Image example](https://user-images.githubusercontent.com/18037362/140496684-e886fc00-612d-44dd-a6fe-c0d47988246f.gif)

## Pre-requisites

1. Purchase a DepthAI (or OAK) model (see [shop.luxonis.com](https://shop.luxonis.com/))

2. Download sample videos
   ```
   python3 download.py
   ```
3. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
python3 main.py [options]
```

Options:

* -t, --threshold: Box confidence threshold. Default: *0.2*

[中文文档](README.zh-CN.md)

# Gen2 License Plates Recognition

This example demonstrates how to run 2 stage inference on DepthAI using Gen2 Pipeline Builder.
First, a license plate is detected on the image and then the cropped license frame is sent to text detection network, 
which tries to decode the license plates texts


> :warning: **This demo is uses an object detector that is trained to detect and recognize chinese license plates**! It may not work with other license plates.  See roboflow's tutorial on training for other regions' plates, here: [https://blog.roboflow.com/oak-deploy-license-plate/](https://blog.roboflow.com/oak-deploy-license-plate/)

## Demo

[![Gen2 License Plates recognition](https://user-images.githubusercontent.com/5244214/111202991-c62f3980-85c4-11eb-8bce-a3c517abeca1.gif)](https://www.youtube.com/watch?v=tB_-mVVNIro "License Plates recognition on DepthAI")

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (conflicts with -cam)
```

To use with a video file, run the script with the following arguments

```
python3 main.py -vid ./chinese_traffic.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 

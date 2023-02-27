[中文文档](README.zh-CN.md)

# License Plates Recognition

This example demonstrates how to run 2 stage inference on DepthAI using Gen2 Pipeline Builder.
First, a license plate is detected on the image and then the cropped license frame is sent to text detection network, 
which tries to decode the license plates texts


> :warning: **This demo is uses an object detector that is trained to detect and recognize chinese license plates**! It may not work with other license plates.  See roboflow's tutorial on training for other regions' plates, here: [https://blog.roboflow.com/oak-deploy-license-plate/](https://blog.roboflow.com/oak-deploy-license-plate/)

Take a look at [How to Train and Deploy a License Plate Detector to the Luxonis OAK](https://blog.roboflow.com/oak-deploy-license-plate/) tutorial for training a custom detector using the Roboflow platform.

## Demo

[![Gen2 License Plates recognition](https://user-images.githubusercontent.com/5244214/111202991-c62f3980-85c4-11eb-8bce-a3c517abeca1.gif)](https://www.youtube.com/watch?v=tB_-mVVNIro "License Plates recognition on DepthAI")

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
# For DepthAI API
python3 main.py [options]
python3 main.py -vid ./chinese_traffic.mp4 # Use video file
python3 main.py -cam # Use DepthAI RGB camera

# For DepthAI SDK
python3 main.py
```

Options:
```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (conflicts with -cam)
```


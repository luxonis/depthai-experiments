# License Plates Recognition

This example demonstrates how to run 2 stage inference on DepthAI
First, a license plate is detected on the image and then the cropped license frame is sent to text detection network, 
which tries to decode the license plates texts

> :warning: **This demo is uses an object detector that is trained to detect and recognize chinese license plates**! It may not work with other license plates.  See roboflow's tutorial on training for other regions' plates, here: [https://blog.roboflow.com/oak-deploy-license-plate/](https://blog.roboflow.com/oak-deploy-license-plate/)

Take a look at [How to Train and Deploy a License Plate Detector to the Luxonis OAK](https://blog.roboflow.com/oak-deploy-license-plate/) tutorial for training a custom detector using the Roboflow platform.

## Demo

[![License Plates recognition](https://user-images.githubusercontent.com/5244214/111202991-c62f3980-85c4-11eb-8bce-a3c517abeca1.gif)](https://www.youtube.com/watch?v=tB_-mVVNIro "License Plates recognition on DepthAI")

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)
```

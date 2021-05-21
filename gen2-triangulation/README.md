[中文文档](README.zh-CN.md)

# Gen2 Triangulation 3D visualizer

Because there are often application-specific host-side filtering to be done on the stereo
neural inference results, and because these calculations are lightweight
(i.e. could be done on an ESP32), we leave the triangulation itself to the host.

This 3D visualizer is for the facial landmarks demo, and uses OpenGL and OpenCV.
Consider it a draft/reference at this point.

## Demo

[![Spatial AI](https://user-images.githubusercontent.com/18037362/116149182-bc2b4b00-a6d9-11eb-91a5-ad5359ca85ad.gif)](https://www.youtube.com/watch?v=YalHMcsZODs&feature=youtu.be "3D Facial Landmark visualization")

## Installation

```
sudo apt-get install python3-pygame
python3 -m pip install -r requirements.txt
```

Note that this experiment uses the `Script` node that is currently in alpha mode, so you have to install the latest `gen2-scripting` branch of the library (you get it by installing `requirements.txt`)

## Usage

Run the application

```
python3 main.py
```

You should see 5 windows appear:
- `mono_left` which will show camera output from left mono camera + face bounding box & facial landmarks
- `mono_right` which will show camera output from right mono camera + face bounding box & facial landmarks
- `crop_left` which will show 48x48 left cropped image that goes into the second NN + facial landmarsk that get outputed from the second NN
- `crop_right` which will show 48x48 right cropped image that goes into the second NN + facial landmarsk that get outputed from the second NN
- `pygame window` which will show the triangulation results

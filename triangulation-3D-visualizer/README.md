[中文文档](README.zh-CN.md)

# Triangulation 3D visualizer

because there are often application-specific host-side filtering to be done on the stereo 
neural inference results, and because these calculations are lightweight 
(i.e. could be done on an ESP32), we leave the triangulation itself to the host.  
If there is interest to do this on DepthAI directly instead, please let us know!

This 3D visualizer is for the facial landmarks demo, and uses OpenGL and OpenCV.  
Consider it a draft/reference at this point.  


## Demo

[![Spatial AI](https://user-images.githubusercontent.com/5244214/90748450-7c9e9a80-e2d2-11ea-9e9e-da65b5d9e6f0.gif)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")

Please that when we recorded this we accidentally inverted the results - so the mouth keypoints are up top, and the eyes on the bottom.


## Installation

```
sudo apt-get install python3-pygame
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

You should see 5 windows appear:
- `previewout-left` which will show camera output from left mono camera
- `previewout-right` which will show camera output from right mono camera
- `left` which will show neural network results based on left mono frame
- `right` which will show neural network results based on right mono frame
- `pygame window` which will show the triangulation results  

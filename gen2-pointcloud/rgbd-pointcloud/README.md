## RGB-D projection Demo

This example shows how to align depth to rgb camera frame and project depth map into 3D pointcloud. You can also choose to skip rgb-depth alignment and colorize the pointcloud with right mono frame.

## Demo

![img](https://user-images.githubusercontent.com/18037362/158274860-efae7bda-88d1-43be-8d50-6b63e023f964.png)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```
Note: Running the command above also tries to install open3D which is required for this example.
open3D is not supported by all platforms, but is required for pointcloud visualization.

## Run this example

```
python3 main.py
```



## TODO:
- Fix z-axis

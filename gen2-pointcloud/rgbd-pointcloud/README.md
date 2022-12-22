## RGB-D projection Demo

This example shows how to align depth to rgb camera frame and project depth map into 3D pointcloud. You can also choose to skip rgb-depth alignment and colorize the pointcloud with right mono frame.

## Demo

![img](https://user-images.githubusercontent.com/18037362/158277114-f1676487-e214-4872-a1b3-aa14131b666b.png)

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

Press s to save the most recent depth/rgb/pcl to disk
Press q to quit

## TODO:
- Fix z-axis

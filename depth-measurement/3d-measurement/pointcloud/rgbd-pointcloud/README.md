## RGB-D projection Demo

This example shows how to align depth to rgb camera frame and project depth map into 3D pointcloud. You can also choose to skip rgb-depth alignment and colorize the pointcloud with right mono frame.

## Demo

![img](https://user-images.githubusercontent.com/18037362/158277114-f1676487-e214-4872-a1b3-aa14131b666b.png)

## Installation

```
python3 -m pip install -r requirements.txt
```

Running the command above also tries to install open3D which is required for this example.
open3D is not supported by all platforms, but is required for pointcloud visualization. Installing open3D on [Python 3.12 is not yet supported](https://stackoverflow.com/questions/62352767/cant-install-open3d-libraries-errorcould-not-find-a-version-that-satisfies-th).

Warning: The experiment requires numpy version 1.\*, numpy 2 is not supported due to open3d incompatibility.

## Usage

Run the application

```
python3 main.py
```

```
optional arguments:
  -h, --help            show this help message and exit
  -m, --mono            use mono frame for pointcloud coloring instead of color frame
```

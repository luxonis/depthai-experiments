## Point cloud fusion
This example demonstrates how point clouds from different cameras can be merged together.

_For a single camera version check out [rgbd-pointcloud](https://github.com/luxonis/depthai-experiments/tree/master/gen2-pointcloud/rgbd-pointcloud)._

![demo](img/demo.gif)
## Install project requirements
```
python3 -m pip install -r requirements.txt
```
> Note: Running the command above also tries to install open3D which is required for this example. Open3D is not supported by all platforms, but is required for pointcloud visualization.

## Usage
> Before you can run this demo you need to calibrate the cameras. Go to [multi-cam-calibration](../multi-cam-calibration) and generate a calibration file for each camera. Make sure that the `calibration_data_dir` in the [`config.py`](config.py) is set correctly.

Run the [`main.py`](main.py) with Python 3.
```
python3 main.py
```



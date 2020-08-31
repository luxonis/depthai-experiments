# Point Cloud Visualizer

This experiment allows you to run and visualize the point clouds generated using depth map and right camera.


## Install
```
python3 -m pip install -r requirements.txt
```

## Steps to get point cloud from depth map and visualize.

- Replace depthai.py, calibrate.py under depthai folder and projector_3d.py, calibration_utils.py under depthai_helpers folder in [depthai](https://github.com/luxonis/depthai/tree/master).
- if dataset folder is previously generated using calibrate.py execute the following command in commmand line inside the depthai folder.
```
python3 calibrate.py -s [SQUARE_SIZE_IN_CM] -b [baseline] -m process
```
- if dataset folder doesn't exist. Calibrate your camera as shown [here](https://docs.luxonis.com/tutorials/stereo_calibration/).
-  Now execute the depthai.py as with `right` and `depth_raw` streams enabled.

:::info
if you don't need point cloud visualization but want to receive only point clouds set `PointCloudVisualizer(file_path, enableViz=false)`
:::

# Sample video
![](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)

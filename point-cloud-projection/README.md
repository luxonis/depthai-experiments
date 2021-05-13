[中文文档](README.zh-CN.md)

# Gen1 Point Cloud Visualizer

This experiment allows you to run and visualize the point clouds generated using depth map and right camera.
# Sample video
![point cloud visualization](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)



# Demo
![demogif](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)

## Installation
```
python3 install_requirements.py
```
Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. This example contains open3d lib which is necessary for point cloud visualization and transformation. However, this library's binaries are not available for some hosts like raspberry pi and jetson. 

## Usage
Run the application
```
python3 main.py
```

> Note: if you don't need point cloud visualization but want to receive only point clouds set `PointCloudVisualizer(file_path, enableViz=false)`


## Calibrate camera (if needed)

To run this application, EEPROM needs to be programmed on your device. Most of the devices we ship has already programmed EEPROM.

If you received the EEPROM error, like the one below:

```
legacy, get_right_intrinsic() is not available in version -1
recalibrate and load the new calibration to the device. 
```

please check application logs for `EEPROM data:` line.

Correct EEPROM data should look like this:

```
EEPROM data: valid (v5)
  Board name     : BW1098OBC
  Board rev      : R0M0E0
  HFOV L/R       : 71.86 deg
  HFOV RGB       : 68.7938 deg
  L-R   distance : 7.5 cm
  L-RGB distance : 3.75 cm
  L/R swapped    : yes
  L/R crop region: center
  Rectification Rotation R1 (left):
    0.999939,   -0.008194,   -0.007420,
    0.008200,    0.999966,    0.000815,
    0.007413,   -0.000876,    0.999972,
  Rectification Rotation R2 (right):
    0.999991,   -0.004153,    0.001323,
    0.004155,    0.999991,   -0.000824,
   -0.001320,    0.000830,    0.999999,
  Calibration intrinsic matrix M1 (left):
  863.789124,    0.000000,  644.251648,
    0.000000,  862.789490,  414.100494,
    0.000000,    0.000000,    1.000000,
  Calibration intrinsic matrix M2 (right):
  862.911011,    0.000000,  626.155884,
    0.000000,  861.774353,  400.832611,
    0.000000,    0.000000,    1.000000,
  Calibration rotation matrix R:
    0.999954,   -0.004038,   -0.008737,
    0.004053,    0.999990,    0.001675,
    0.008730,   -0.001711,    0.999960,
  Calibration translation matrix T:
   -7.510510,
    0.031195,
   -0.009940,
  Calibration Distortion Coeff d1 (Left):
   14.766488,  -23.925636,    0.001765,   -0.001056,  180.061539,   14.801581,  -25.798475,
  184.282516,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
  Calibration Distortion Coeff d2 (Right):
   -5.345910,   19.003929,    0.002202,   -0.000723,  -21.828526,   -5.399135,   19.195726,
  -22.006199,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
```

But there is possibility that your board will show this instead

```
EEPROM data: invalid / unprogrammed
```

Or that the programmed EEPROM is old and does not contain intrinsic matrices in it

```
EEPROM data: valid (v3)
  Board name     : BW1098OBC
  Board rev      : R0M0E0
  HFOV L/R       : 71.86 deg
  HFOV RGB       : 68.7938 deg
  L-R   distance : 7.5 cm
  L-RGB distance : 3.75 cm
  L/R swapped    : yes
  L/R crop region: center
  Calibration homography:
    1.000510,   -0.008015,    3.452457,
    0.010204,    0.997410,   -6.814469,
    0.000006,    0.000000,    1.000000,
```

In this case, you need to recalibrate the camera and store a new EEPROM on the device.

To do so, please follow [calibration tutorial](https://docs.luxonis.com/products/stereo_camera_pair/#stereo-calibration)
and then store the new calibration into EEPROM run:

```
# in depthai repository
python3 depthai_demo.py --store-eeprom -brd bw1098obc  # replace "bw1098obc" if using another board 
```

Now run the script again and check if the EEPROM data contains correct intrinsic information

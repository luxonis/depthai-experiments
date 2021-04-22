[英文文档](README.md)

# 范例视频
![point cloud visualization](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)

# 点云可视化器

通过此实验，您可以运行和可视化使用深度图和右摄像头生成的点云。

# 演示
![demogif](https://media.giphy.com/media/W2Es1aC7N0XZIlyRmf/giphy.gif)

## 安装依赖
```
python3 install_requirements.py
```
注意: `python3 install_requirements.py` 还会尝试从requirements-optional.txt中安装可选的库。此示例包含open3d lib，这对于点云可视化和转换是必需的。但是，该库的二进制文件不适用于树莓派和jetson之类的某些主机。

## 用法
运行应用程序
```
python3 main.py
```

> 注意: 如果您不需要点云可视化，但只想接收设置的点云 `PointCloudVisualizer(file_path, enableViz=false)`


## 校准相机（如果需要）

要运行该应用程序，需要在设备上对EEPROM进行编程。我们出厂的大多数设备都已对EEPROM进行了编程。

如果您收到EEPROM错误，例如以下错误:

```
legacy, get_right_intrinsic() is not available in version -1
recalibrate and load the new calibration to the device. 
```

请检查应用程序日志中是否有 `EEPROM data:` 行.

正确的EEPROM数据应如下所示:

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

但是您的董事会有可能会改为显示此内容

```
EEPROM data: invalid / unprogrammed
```

或者编程的EEPROM过旧并且其中不包含固有矩阵

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

在这种情况下，您需要重新校准相机并将新的EEPROM存储在设备上。

为此，请遵循 [校准教程](https://docs.luxonis.com/products/stereo_camera_pair/#stereo-calibration)
然后将新的校准存储到EEPROM运行中:

```
# in depthai repository
python3 depthai_demo.py --store-eeprom -brd bw1098obc  # replace "bw1098obc" if using another board 
```

现在再次运行脚本，并检查EEPROM数据是否包含正确的内部信息。
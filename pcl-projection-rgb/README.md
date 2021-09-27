[中文文档](README.zh-CN.md)

# Gen1 RGB-D convertion on Host

# Sample video
![rgb-d](https://media.giphy.com/media/SnW9p4r3feMQGOmayy/giphy.gif)

# RGB-D conversion

In this experiment `rgbd_creating_o3d.py/rgbd_creating_no_o3d.py` allows you to convert depth in rectified_right frame to rgb camera frame

`rgbd_creating_no_o3d.py` will have some noise.

# Point cloud with rgb 
![demogif](https://media.giphy.com/media/UeAlkPpeHaxItO0NJ6/giphy.gif)


 - Use `colorized_pont_cloud.py` to obtain point could in rgb camera reference frame with color.(if you don't need color overlapped with rgb you can skip 2 steps)

## Installation

```
python3 install_requirements.py
```
Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. This example contains open3d lib which is necessary for point cloud visualization and transformation. However, this library's binaries are not available for some hosts like raspberry pi and jetson. 
In times where open3D is not supported on your host. try  `rgbd_creating_no_o3d.py`. This program works independent of open3D.


## Calibrate camera (if needed)

To run this application, your device needs to be calibrated with rgb camera which was not carried out in devices before Dec 2020. Will soon provide an update new calibration tool to obtain rgb camera calibration

If you received the EEPROM error, like the one below:



```
legacy, get_right_intrinsic() is not available in version -1
recalibrate and load the new calibration to the device. 
```


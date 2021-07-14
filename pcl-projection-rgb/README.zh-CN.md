[英文文档](README.md)

# 范例视频
![rgb-d](https://media.giphy.com/media/SnW9p4r3feMQGOmayy/giphy.gif)

# RGB-D 转换

在此实验中 `rgbd_creating_o3d.py/rgbd_creating_no_o3d.py` 您可以将rectified_right帧中的深度转换为RGB摄像机帧


`rgbd_creating_no_o3d.py` 会包含一些噪音。

# 点云与RGB
![demogif](https://media.giphy.com/media/UeAlkPpeHaxItO0NJ6/giphy.gif)

用于 `colorized_point_cloud.py` 获取在RGB相机参考帧中具有颜色的点。（如果不需要与RGB重叠的颜色，则可以跳过2步）

## 安装依赖

```
python3 install_requirements.py
```
注意: `python3 install_requirements.py` a还会尝试从requirements-optional.txt中安装可选的库。此示例包含open3d lib，这对于点云可视化和转换是必需的。但是，该库的二进制文件不适用于树莓派和jetson之类的某些主机。在主机不支持open3D的情况下。试一试  `rgbd_creating_no_o3d.py`. 该程序与open3D无关。

## 校准相机（如果需要）

要运行此应用程序，您的设备需要使用RGB相机进行校准，而2020年12月之前该设备尚未在设备中进行过校准。将很快提供更新的新校准工具来获取RGB相机校准

如果您收到EEPROM错误，例如以下错误:

```
legacy, get_right_intrinsic() is not available in version -1
recalibrate and load the new calibration to the device. 
```


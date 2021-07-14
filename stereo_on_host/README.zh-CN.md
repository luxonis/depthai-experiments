[英文文档](README.md)

# 视差深度 RGB right
本示例使用右摄像头和RGB摄像头创建视差，但是此示例希望您的RGB已校准

## 安装依赖

```
python3 -m pip install -r requirements.txt
```

## 校准相机（如果需要）

要运行此应用程序，您的设备需要使用RGB相机进行校准，而2020年12月之前该设备尚未在设备中进行过校准。将很快提供更新的新校准工具来获取RGB相机校准

如果您收到EEPROM错误，例如以下错误:

```
legacy, get_right_intrinsic() is not available in version -1
recalibrate and load the new calibration to the device. 
```


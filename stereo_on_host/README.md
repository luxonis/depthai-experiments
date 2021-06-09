[中文文档](README.zh-CN.md)

# Gen1 Stereeo rgb-right
This example uses right camera and the rgb camera to create disparity.However this example expects that your rgb is calibraated


## Installation

```
python3 -m pip install -r requirements.txt
```

## Calibrate camera (if needed)

To run this application, your device needs to be calibrated with rgb camera which was not carried out in devices before Dec 2020. Will soon provide an update new calibration tool to obtain rgb camera calibration

If you received the EEPROM error, like the one below:

```
legacy, get_right_intrinsic() is not available in version -1
recalibrate and load the new calibration to the device. 
```


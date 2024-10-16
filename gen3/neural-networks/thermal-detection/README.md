## Thermal detection

This pipeline implements people and vehicle detection model on thermal camera output. The detection model is our version of YOLOv6 architecture and we used [FLIR](https://www.flir.com/oem/adas/adas-dataset-form/?srsltid=AfmBOor3VUq0IAfuYxNlTEAbjJUTxTJ6Zmlh0xybNWpUKueEFfEKe9xH) and other publicly available datasets to train on. For inference we use `color` output from the thermal camera which we additionaly convert from YUV to BGR to be able to feed to the model.

## Installation
```
python3 -m pip install -r requirements.txt
```

Note: You might need to install correct version of depthai from the [depthai-core](https://github.com/luxonis/depthai-core) directly if it is not yet publicly available.

## Usage
Run the application

```
python3 main.py
```
or

```
python3 main.py --video_path <path to mp4>
```




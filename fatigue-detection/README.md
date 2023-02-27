[中文文档](README.zh-CN.md)

# Fatigue Detection

This example demonstrates the Gen2 Pipeline Builder running [face detection network](https://docs.openvinotoolkit.org/2019_R1/_face_detection_retail_0004_description_face_detection_retail_0004.html) and head detection network

## Demo:

![Fatigue detection](assets/fatigue.gif)

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
# For DepthAI API
python3 main.py [options]

# For DepthAI SDK
python3 main.py
```

Options:
```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
   -h, --help            show this help message and exit
   -nd, --no-debug       Prevent debug output
   -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
   -vid VIDEO, --video VIDEO
                           Path to video file to be used for inference (conflicts with -cam)
```
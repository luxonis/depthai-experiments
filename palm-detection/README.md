[中文文档](README.zh-CN.md)

Palm detection
================

This example demonstrates the Gen2 Pipeline Builder running 
[palm detection network](https://google.github.io/mediapipe/solutions/hands#palm-detection-model)  

## Demo:

![demo](assets/palm_detection.gif)
--------------------

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
# DepthAI API
python3 main.py [options]
python3 main.py -vid <path> # Use video file
python3 main.py -cam # Use DepthAI RGB camera

# DepthAI SDK
python3 main.py
```

Options:
```bash
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        The path of the video file used for inference (conflicts with -cam)

```

> Press 'q' to exit the program.

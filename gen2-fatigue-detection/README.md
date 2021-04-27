[中文文档](README.zh-CN.md)

# Gen2 Fatigue Detection

This example demonstrates the Gen2 Pipeline Builder running [face detection network](https://docs.openvinotoolkit.org/2019_R1/_face_detection_retail_0004_description_face_detection_retail_0004.html) and head detection network

## Demo:

![Fatigue detection](media/fatigue.gif)

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
```
python -m pip install -r requirements.txt
```

## Usage

```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
   -h, --help            show this help message and exit
   -nd, --no-debug       Prevent debug output
   -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
   -vid VIDEO, --video VIDEO
                           Path to video file to be used for inference (conflicts with -cam)
```

### Run the program using the device

```
python main.py -cam
```

### Run the program using video
   
```   
python main.py -vid <path>
```

Press 'q' to exit the program.
Palm detection
================

This example demonstrates the Gen2 Pipeline Builder running 
[palm detection network](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)  

## Demo:

![demo](images/palm_detection.gif)
--------------------

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        The path of the video file used for inference (conflicts with -cam)

```

To use with a video file, run the script with the following arguments

```
python main.py -vid <path>
```

To use with DepthAI 4K RGB camera, use instead
```
python main.py -cam
```

[中文文档](README.zh-CN.md)

Fire detection
================

This example demonstrates the Gen2 Pipeline Builder running [fire detection network](https://github.com/StephanXu/FireDetector/tree/python)  

## Demo:

![demo](images/fire_demo.gif)
--------------------

## Pre-requisites

Install requirements:
```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       prevent debug output
  -vid VIDEO, --video VIDEO
                        The path of the video file used for inference (otherwise uses DepthAI 4K RGB camera)

```

To use with a video file, run the script with the following arguments

```bash
python main.py -vid <path>
```

To use with DepthAI 4K RGB camera, use instead
```bash
python main.py
```

> Press 'q' to exit the program.

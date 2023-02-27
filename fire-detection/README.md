[中文文档](README.zh-CN.md)

Fire detection
================

This example demonstrates the Gen2 Pipeline Builder running [fire detection network](https://github.com/StephanXu/FireDetector/tree/python)  

## Demo:

![demo](api/images/fire_demo.gif)
--------------------

## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py [options]
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

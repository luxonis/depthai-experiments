[中文文档](README.zh-CN.md)

# Gen2 NN frame sync

This example shows how to present the neural network inference results on the inferenced frames.
Two ways of frame sync were used:
- Device sync, using DepthAI device-side queues (results of face detections are presented this way)
- Host sync, using built-in `queue` module  (results of landmarks detection are presented this way)

## Demo

![image](https://user-images.githubusercontent.com/5244214/104956823-36f31480-59cd-11eb-9568-64c0f0003dd0.gif)


## Setup

```
python3 -m pip -U pip
python3 -m pip install -r requirements.txt
```

## Run

```
python3 main.py
```
[中文文档](README.zh-CN.md)

## Deeplabv3 person segmentation on DepthAI

This example shows how to run Deeplabv3+ with DepthAI API.

[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/109359126-25a9ed00-7842-11eb-9071-cddc7439e3ca.png)](https://www.youtube.com/watch?v=zjcUChyyNgI "Deeplabv3+ Custom Training for DepthAI")

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py --size [256,513]
```

- Size can be either 256 or 513. 256x256 model is smaller and faster, while 513x513 is slower and more accurate.


[中文文档](README.zh-CN.md)

## Deeplabv3 person segmentation on DepthAI

This example shows how to run Deeplabv3+ with DepthAI API.

[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/109359126-25a9ed00-7842-11eb-9071-cddc7439e3ca.png)](https://www.youtube.com/watch?v=zjcUChyyNgI "Deeplabv3+ Custom Training for DepthAI")

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
python3 main.py --size [256,513]

# For DepthAI SDK
python3 main.py
```
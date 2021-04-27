[中文文档](README.zh-CN.md)

# Two Stage Inference

This eample shows how to perform two stage inference using DepthAI.

We'll use `face-detection-retail-0004` to detect face and `landmarks-regression-retail-0009` as 
second stage inference which will detect facial landmarks on the detected face

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
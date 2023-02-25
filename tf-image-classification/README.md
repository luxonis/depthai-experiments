[中文文档](README.zh-CN.md)

# Tensorflow Image Classification example

This example demonstrates how to run a neural network created using [TensorFlow Image Classification tutorial](https://colab.research.google.com/drive/1oNxfvx5jOfcmk1Nx0qavjLN8KtWcLRn6?usp=sharing)
(which one of our community members has put together in a single Colab Notebook, even with OpenVINO conversion to .blob)


## Demo

![Pedestrian Re-Identification](https://user-images.githubusercontent.com/5244214/109003919-522a0180-76a8-11eb-948c-a74432c22be1.gif)

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
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

To use with a video file, run the script with the following arguments

```
python3 main.py -vid ./input.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 

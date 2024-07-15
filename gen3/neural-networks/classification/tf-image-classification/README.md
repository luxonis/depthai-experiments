## Gen2 Tensorflow Image Classification example

This example demonstrates how to run a neural network created using [TensorFlow Image Classification tutorial](https://colab.research.google.com/drive/1oNxfvx5jOfcmk1Nx0qavjLN8KtWcLRn6?usp=sharing)
(which one of our community members has put together in a single Colab Notebook, even with OpenVINO conversion to .blob)


## Demo

![Flower identification](https://user-images.githubusercontent.com/5244214/109003919-522a0180-76a8-11eb-948c-a74432c22be1.gif)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)
```

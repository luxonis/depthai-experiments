## Full image classification

This example demonstrates how to do classification on a full image using a neural network.

# Efficientnet-b0 classification model

You can read more about the EfficientDet model in [OpenVINO's Docs](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientnet-b0)
This model includes 1000 classes of image classification. You can view these classes in the `efficientnet_classes.py` file.

# TensorFlow imgage classification

This neural network was created using [TensorFlow Image Classification tutorial](https://colab.research.google.com/drive/1oNxfvx5jOfcmk1Nx0qavjLN8KtWcLRn6?usp=sharing)
(which one of our community members has put together in a single Colab Notebook, even with OpenVINO conversion to .blob)

## Demo

The efficientnet demo classifies the animals in the images as Ibex (Mountain Goat) accurately. Also classifies most common objects.

![Animal classification](https://user-images.githubusercontent.com/67831664/119170640-2b9a1d80-ba81-11eb-8a3f-a3837af38a73.jpg)

The tensorflow demo was trained on flower classes.

![Flower classification](https://user-images.githubusercontent.com/5244214/109003919-522a0180-76a8-11eb-948c-a74432c22be1.gif)

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
  -nn {efficientnet, flowers}, --neural-network {efficientnet, flowers}
                        Choose the neural network model used for classification (efficientnet is default)
  -vid VIDEO_PATH, --video VIDEO_PATH
                        Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)
```

## Car detection with YoloV6-nano on DepthAI 

This example shows how to run YoloV6-nano object detection on DepthAI.

The following *.blob*s are available:

You can find the tutorial for training the model and generation of *.blob* file [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks) - **YoloV6_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV6_training.ipynb). You can create a copy of the Colab Notebook and try training the model on your own! 

See the example of a model pre-trained on [*Annotated driving dataset*](https://github.com/udacity/self-driving-car/tree/master/annotations) from Udacity data set:

![Example](https://user-images.githubusercontent.com/56075061/143061151-07157024-4189-420d-b603-2cb3ec926bf5.png)

Source Image: [Pexels](https://www.pexels.com/video/different-kinds-of-vehicles-on-the-freeway-2053100/)

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```
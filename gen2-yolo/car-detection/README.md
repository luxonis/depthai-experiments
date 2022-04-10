## Car detection with YoloV3-tiny or YoloV4-tiny on DepthAI 

This example shows how to run YoloV3-tiny and YoloV4-tiny object detection on DepthAI, and is the **"next steps"** repository for the **YoloV3_V4_tiny_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb) tutorial.

The following *.blob*s are available:

* *yolo_v3_tiny_openvino_2021.3_6shave.blob*: YoloV3-tiny version, and
* *yolo_v4_tiny_openvino_2021.3_6shave.blob*: YoloV4-tiny version.

You can find the tutorial for training the model and generation of *.blob* file [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks) - **YoloV3_V4_tiny_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb). You can create a copy of the Colab Notebook and try training the model on your own!

See the example of a model pre-trained on [*Annotated driving dataset*](https://github.com/udacity/self-driving-car/tree/master/annotations) from Udacity data set:

![Example](https://user-images.githubusercontent.com/56075061/143061151-07157024-4189-420d-b603-2cb3ec926bf5.png)

Source Image: [Pexels](https://www.pexels.com/video/different-kinds-of-vehicles-on-the-freeway-2053100/)

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

Add your blobs to `models` directory, or download pretrained models using:
```
python3 download.py
```
## Usage

```
python3 main.py -m <relative_path_to_blob>
```

## Options

You can edit the `yolo-tiny.json` and set it up for your model, as described in the **next steps** section in the [Colab](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb). In case you are changing some of the parameters in the tutorial, you should edit the corresponding parameters. JSON is currently set to the values used in training and is the same for both, YoloV3-tiny and YoloV4-tiny car detection models.

**Note**: Values must match the values set in the CFG during training. If you use different input width, you should also change `side32` to `sideX` and `side16` to `sideY`, where `X = width/16` and `Y = width/32`.

You can also change IOU and confidence thresholds.

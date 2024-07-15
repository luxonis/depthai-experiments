## YoloV5 decoding on host

This example shows how to run YoloV5 object detection on DepthAI with decoding on host. We also provide support for on-device decoding. Detailed steps are available in the **YoloV5_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb) tutorial.

The following *.blob*s are available:

* *yolov5s_sku_openvino_2021.4_6shave.blob*: pretrained on YoloV5s on 3 epochs of SKU-110K dataset as can also be seen in the Colab.
* *yolov5s_default_openvino_2021.4_6shave.blob*: pretrained version of YoloV5s on Coco dataset.
* *yolov5m_default_openvino_2021.4_6shave*: heavier version (YoloV5m) pretrained on Coco dataset.

For differences between YoloV5s and YoloV5m please refer to the [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

You can find the tutorial for training the model and generation of *.blob* file [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks) - **YoloV5_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb). You can create a copy of the Colab Notebook and try training the model on your own!

See the example of a model pretrained on SKU-110K dataset in action:

![Example Image](https://user-images.githubusercontent.com/56075061/145186805-38e3115d-94fa-4850-9ec4-c34f90c05d30.gif)

## Pre-requisites

1. Install requirements:
   ```
   python3 -m pip install -r requirements.txt
   ```
2. Download models
   ```
   python3 download.py
   ```

## Usage

```
python3 main.py [options]
```

Options:

* -cam, --cam_input: Select camera input source for inference. Available options: left, right, rgb (default).
* -nn, --nn_model: Select model path for inference. Default: *models/yolov5s_sku_openvino_2021.4_6shave.blob*
* -conf, --confidence_thresh: Set the confidence threshold. Default: 0.3.
* -iou, --iou_thresh: Set the NMS IoU threshold. Default: 0.4.

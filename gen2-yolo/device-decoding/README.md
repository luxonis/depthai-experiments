## Yolo detection with on-device decoding

This repository contains the code based on DepthAI API (`main.py`) and DepthAI SDK (`main_sdk.py`) and shows how to run a Yolo object detector with on-device decoding. Currently, the supported versions are:

* YoloV3 & YoloV3-tiny,
* YoloV4 & YoloV4-tiny,
* YoloV5.

### Export your model

As the models have to be exported to OpenVINO IR in a certain way, we provide the tutorials on training and exporting:

* YoloV3, YoloV4, and their tiny versions: **YoloV3_V4_tiny_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb),

* YoloV5: **YoloV5_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb)

## Usage

```python
python3 main.py -m <model_name> -c <config_json>
```
where:

* `<model_name>` is the **name of the model** from DepthAI model zoo (https://zoo.luxonis.com) or from OpenVINO model ZOO or **relative path to the blob** file.
* `<config_json>` is the **relative path** to the JSON with metadata about the Yolo model.

## JSONs

You can edit the `yolo-tiny.json` and set it up for your model, as described in the **next steps** section in the [Colab](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb). In case you are changing some of the parameters in the tutorial, you should edit the corresponding parameters.

**Note**: Values must match the values set in the CFG during training. If you use different input width, you should also change `side32` to `sideX` and `side16` to `sideY`, where `X = width/16` and `Y = width/32`.

You can also change IOU and confidence thresholds.

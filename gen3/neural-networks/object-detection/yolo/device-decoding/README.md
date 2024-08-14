# Yolo detection with on-device decoding

![Yolo-on-device](https://user-images.githubusercontent.com/56075061/144863222-a52be87e-b1f0-4a0a-b39b-f865bbb6e4a4.png)

This repository contains the code for running Yolo object detection with on-device decoding with [DepthAI API](https://docs.luxonis.com/projects/api/en/latest/) (`main.py`). Currently, supported versions are:

* YoloV6

### Export your model

As the models have to be exported to OpenVINO IR in a certain way, we provide the tutorials on training and exporting:

* Yolo**V6**: *YoloV6_training.ipynb* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV6_training.ipynb).


## Usage

1. Install requirements
	```python
	python3 -m pip install -r requirements.txt
	```
2. Run the script
    ```
    python3 main.py
    ```

## Depth information

DepthAI enables you to take the advantage of depth information and get `x`, `y`, and `z` coordinates of detected objects. Experiments in this directory are not using the depth information. If you are interested in using the depth information with Yolo detectors, please check our [documentation](https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_tiny_yolo/#rgb-tinyyolo-with-spatial-data).

![SpatialObjectDetection](https://user-images.githubusercontent.com/56075061/144864639-4519699e-d3da-4172-b66b-0495ea11317e.png)

# Yolo detection with on-device decoding

![Yolo-on-device](https://user-images.githubusercontent.com/56075061/144863222-a52be87e-b1f0-4a0a-b39b-f865bbb6e4a4.png)

This repository contains the code for running Yolo object detection with on-device decoding with [DepthAI SDK](https://docs.luxonis.com/projects/sdk/en/latest/) (`main.py`)  or [DepthAI API](https://docs.luxonis.com/projects/api/en/latest/) (`main_api.py`) directly. Currently, supported versions are:

* YoloV3 & YoloV3-tiny
* YoloV4 & YoloV4-tiny
* YoloV5
* YoloV6
* YoloV7 (detections only)

We use the same style of JSON parsing in `main.py` and in `main_api.py`, but you can also set the values in both cases manually in the code.


## Yolo with DepthAI SDK tutorial

I have trained my own object detection model by following the
[YoloV6_training.ipynb](https://github.com/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV6_training.ipynb) notebook. After downloading the `.pt` file from the notebook,
I uploaded it to [tools.luxonis.com](https://tools.luxonis.com), specified the model shape to `640 352` and downloaded the .zip file. After extracting zip into the `model` folder, the json config file was at `./model/yolov6n.json`:

```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        12/27/2022  10:53 PM             28 .gitignore
-a----        12/28/2022   3:34 PM       18074356 yolov6n-simplified.onnx
-a----        12/28/2022   3:35 PM        8634878 yolov6n.bin
-a----        12/28/2022   3:35 PM           2340 yolov6n.json
-a----        12/28/2022   3:35 PM         163855 yolov6n.xml
-a----        12/28/2022   3:35 PM        8691456 yolov6n_openvino_2021.4_6shave.blob
```

Then I ran `python main.py --config model/yolov6n.json` to run the model inference (object detection) on the OAK camera, and got these two frames:

![image](https://user-images.githubusercontent.com/18037362/209836980-c4ab8ffe-faab-4feb-9a8e-1ba9227f7fd7.png)

To configure the pipeline, you can edit `main.py` script, and SDK documentation can be [found here](https://docs.luxonis.com/projects/sdk/en/latest/).

### Export your model

As the models have to be exported to OpenVINO IR in a certain way, we provide the tutorials on training and exporting:

* Yolo**V3**, Yolo**V4**, and their **tiny** versions: *YoloV3_V4_tiny_training.ipynb* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb),

* Yolo**V5**: *YoloV5_training.ipynb* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb).


## Usage

1. Install requirements
	```python
	python3 -m pip install -r requirements.txt
	```
2. Run the script
    ```
    python3 main.py --config <config_json>
    ```
    or
    ```
    python3 main_api.py -m <model_name> --config <config_json>
    ```
    where:

    * `<model_name>` is the **name of the model** from DepthAI model zoo (https://zoo.luxonis.com) or **relative path to the blob** file. Please check our model zoo to see which pre-trained models are available.
    * `<config_json>` is the **relative path** to the JSON with metadata (input shape, anchors, labels, ...) of the Yolo model.


## JSONs

We already provide some JSONs for common Yolo versions. You can edit them and set them up for your model, as described in the **next steps** section in the mentioned tutorials. In case you are changing some of the parameters in the tutorial, you should edit the corresponding parameters. In general, the settings in the JSON should follow the settings in the CFG of the model. For YoloV5, the default settings should be the same as for YoloV3.

**Note**: Values must match the values set in the CFG during training. If you use a different input width, you should also change `side32` to `sideX` and `side16` to `sideY`, where `X = width/16` and `Y = width/32`. If you are using a non-tiny model, those values are `width/8`, `width/16`, and `width/32`.

You can also change IOU and confidence thresholds. Increase the IOU threshold if the same object is getting detected multiple times. Decrease confidence threshold if not enough objects are detected. Note that this will not magically improve your object detector, but might help if some objects are filtered out due to the threshold being too high.

## Depth information

DepthAI enables you to take the advantage of depth information and get `x`, `y`, and `z` coordinates of detected objects. Experiments in this directory are not using the depth information. If you are interested in using the depth information with Yolo detectors, please check our [documentation](https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_tiny_yolo/#rgb-tinyyolo-with-spatial-data).

![SpatialObjectDetection](https://user-images.githubusercontent.com/56075061/144864639-4519699e-d3da-4172-b66b-0495ea11317e.png)

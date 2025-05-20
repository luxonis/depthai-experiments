# Multi-Input YOLO World Demo README

This example demonstrates the implementation of multi-input [YOLO-World](https://github.com/AILab-CVC/YOLO-World) object detection pipeline on DepthAI.

**NOTE:** This experiment works only on `RVC4` devices and currently only in `PERIPHERAL` mode.

## Demo

![Barrel detection](media/barrel-detection.gif)

## Features

- Detect objects in real-time using YOLO.
- Support for video files and live camera input.
- Customizable class names and confidence threshold.

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment by running:

```bash
pip install -r requirements.txt
```

## Usage

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                     Optional name, DeviceID or IP of the camera to connect to. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                     Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-c CLASS_NAMES [CLASS_NAMES ...], --class_names CLASS_NAMES [CLASS_NAMES ...]
                     Class names to be detected (default: ['person', 'chair', 'TV'])
-conf CONFIDENCE_THRESH, --confidence_thresh CONFIDENCE_THRESH
                     Sets the confidence threshold (default: 0.1)
```

### Example

```bash
python main.py --class_names person car dog --confidence_thresh 0.2
```

This will run the example by detecting `person`, `car` and `dog` classes using 0.2 as confidence threshold.

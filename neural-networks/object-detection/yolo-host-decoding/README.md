# YOLOv6 Nano decoding on host

This example shows how to run [YOLOv6 Nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) object detection on DepthAI with decoding on host. The neural network processes the video stream on-device and sends the raw outputs to the host for decoding. The decoding of YOLO's outputs is done in the host node where final bounding boxes in form of a `ImgDetections` message are created.

Alternatively, you can use fully on-device decoding with `DetectionNetwork`.

You can find the tutorial for training the custom YOLO model and generation of *.blob* file [here](https://github.com/luxonis/depthai-ml-training/tree/main/training/others/object-detection) - **YoloV6_training.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV6_training.ipynb). You can create a copy of the Colab Notebook and try training the model on your own!

## Demo

<!-- ![Example Image](https://user-images.githubusercontent.com/56075061/145186805-38e3115d-94fa-4850-9ec4-c34f90c05d30.gif) -->

![Demo](../../media/yolov6-nano.gif)

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --media <MEDIA> --fps_limit <FPS_LIMIT> -conf <CONF> -iou <IOU>
```

- `<DEVICE>`: Device IP or ID. Default: \`\`.
- `<MEDIA>`: Path to the video file. Default `None` - camera input.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.
- `<CONF>`: Set the confidence threshold. Default: 0.3.
- `<IOU>` : Set the NMS IoU threshold. Default: 0.4.

#### Examples

```bash
python3 main.py
```

This will run the YOLO object detection experiment with the default device and camera input.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the YOLO object detection experiment with the default device and the video file.

```bash
python3 main.py --device <DEVICE IP OR MXID>
```

This will run the YOLO object detection experiment with the specified device.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <device-ip>
oakctl app run .
```

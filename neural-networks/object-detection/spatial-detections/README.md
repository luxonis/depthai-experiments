# Spatial Detections

This experiment shows how to use DepthAI and OAK camera to detect object with their spatial coordinates in real-time! The experiment by default uses [YOLOv6 Nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) object detection model but you can use other models as well by using `-m` or `--model` argument. It should work with any YOLO detection model.

You can read more about spatial detection network in our [documentation](https://docs.luxonis.com/software/depthai-components/nodes/yolo_spatial_detection_network/). It combines the bounding boxes with the depth information into spatial image detections.

## Demo

![Exampe](media/spatial-detections.gif)

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30.0)
-m MODEL, --model MODEL
                    Model reference to use for object detection. (default: luxonis/yolov6-nano:r2-coco-512x288)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the experiment with the default YOLOv6-Nano model.

```bash
python3 main.py --model luxonis/yolov6-large:r2-coco-640x352
```

This will run the experiment with the specified YOLOv6-Large model.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.

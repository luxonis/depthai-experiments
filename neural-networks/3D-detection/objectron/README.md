# Objectron: 3D object detection

This experiment demonstrates how to perform 3D object detection using the [Objectron](https://hub.luxonis.com/ai/models/4c7a51db-9cbe-4aee-a4c1-b8abbbe18c11) model. The model can predict 3D bounding box of the foreground object in the image. For general object detection we use [YOLOv6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model. The pipepile is a standard 2-stage pipeline with detection and 3D object detection models. The experiment works on both RVC2 and RVC4. [Objectron](https://hub.luxonis.com/ai/models/4c7a51db-9cbe-4aee-a4c1-b8abbbe18c11) can predict 3D bounding boxes for chairs, cameras, cups, and shoes.

## Demo

![chair](media/chair.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

The provided experiment detects chairs, but you can change the object class by changing the valid labels in `script_*.py` file and in constructing the `AnnotationNode` in `main.py`: `annotation_node = pipeline.create(AnnotationNode, connection_pairs=connection_pairs, valid_labels=[41])` where `41` is the label for cups.

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30.0)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

**NOTE**: Camera and shoes can not be detected with general YOLOv6 detector. So, you need to provide your own detector for these objects.

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the objectron experiment with the default device and camera input.

```
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the objectron experiment with the default device and the video file.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <device-ip>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.

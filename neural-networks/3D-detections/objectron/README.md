# Objectron: 3D object detection

This experiment demonstrates how to perform 3D object detection using the [Objectron](https://hub.luxonis.com/ai/models/4c7a51db-9cbe-4aee-a4c1-b8abbbe18c11) model. The model can predict 3D bounding box of the foreground object in the image. For general object detection we use [YOLOv6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model. The pipepile is a standard 2-stage pipeline with detection and 3D object detection models. The experiment works on both RVC2 and RVC4. [Objectron](https://hub.luxonis.com/ai/models/4c7a51db-9cbe-4aee-a4c1-b8abbbe18c11) can predict 3D bounding boxes for chairs, cameras, cups, and shoes.

## Demo

<!-- ![chair](https://user-images.githubusercontent.com/60359299/153616321-83f9d7ae-100a-4317-afd4-72bb3d1011d4.gif) -->

![chair](media/chair.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

The provided experiment detects chairs, but you can change the object class by changing the valid labels in `script_*.py` file and in constructing the `AnnotationNode` in `main.py`: `annotation_node = pipeline.create(AnnotationNode, connection_pairs=connection_pairs, valid_labels=[41])` where `41` is the label for cups.

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

**NOTE**: Camera and shoes can not be detected with general YOLOv6 detector. So, you need to provide your own detector for these objects.

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --media <MEDIA> --fps_limit <FPS_LIMIT>
```

- `<DEVICE>`: Device IP or ID. Default: \`\`.
- `<MEDIA>`: Path to the video file. Default `None` - camera input.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

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

and run the example using the `run_standalone.py` script:

```bash
python3 run_standalone.py \
    --device <DEVICE IP> \
    --media <MEDIA> \
    --fps_limit <FPS>
```

The arguments are the same as in the Peripheral mode.

#### Example

```bash
python3 run_standalone.py \
    --fps_limit 20 \
```

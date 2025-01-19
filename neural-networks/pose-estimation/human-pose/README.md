# Human pose estimation

This experiment demonstrates how to build a 2-stage DepthAI pipeline for human pose estimation. The pipeline consists of [YOLOv6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) object detector and [Lite-HRNet](https://hub.luxonis.com/ai/models/c7c9e353-9f6d-43e1-9b45-8edeae82db70) pose estimation model. The experiment works on both RVC2 and RVC4. For realtime application you will need to use OAK4 cameras.

As an alternative you can use the end-to-end [YOLOv8 Nano Pose Estimation](https://hub.luxonis.com/ai/models/12acd8d7-25c0-4a07-9dff-ab8c5fcae7b1) or [YOLOv8 Large Pose Estimation](https://hub.luxonis.com/ai/models/8be178a0-e643-4f1e-b925-06512e4e15c7) models with [generic example](../../../generic-example/).

There are 4 models available for the Lite-HRNet model, check them on [HubAI](https://hub.luxonis.com/ai/models/c7c9e353-9f6d-43e1-9b45-8edeae82db70). If you choose to use a different model please adjust `fps_limit` accordingly.

## Demo

[![Human pose estimation](media/dance.gif)](media/dance.gif)

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --model <MODEL> --media <MEDIA> --fps_limit <FPS_LIMIT>
```

- `<MODEL>`: HubAI Model Reference from Luxonis HubAI. Default: `luxonis/lite-hrnet:18-coco-192x256`.
- `<DEVICE>`: Device IP or ID. Default: \`\`.
- `<MEDIA>`: Path to the video file. Default `None` - camera input.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

#### Examples

```bash
python3 main.py
```

This will run the human pose estimation experiment with the default device, default model, and camera input.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the human pose estimation experiment with the default device and the video file.

```bash
python3 main.py --model luxonis/lite-hrnet:30-coco-192x256 --fps_limit 5
```

This will run the human pose estimation experiment with the default device and camera input, but with the `luxonis/lite-hrnet:30-coco-192x256` model and a `5` FPS limit.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

and run the example using the `run_standalone.py` script:

```bash
python3 run_standalone.py \
    --model <MODEL> \
    --device <DEVICE> \
    --fps_limit <FPS> \
    --media <MEDIA>
```

The arguments are the same as in the Peripheral mode.

#### Example

```bash
python3 run_standalone.py \
    --device <DEVICE IP> \
    --fps_limit 20 \
```

# YOLO-P on DepthAI

This experiment shows an implementation of [YOLO-P](https://hub.luxonis.com/ai/models/0a22d194-d525-46e7-a785-a267b7958a39?view=page) from our HubAI. It shows that YOLO-P can be run as a ADAS (advanced driving assistance system) on DepthAI. It can detect vehicles, segment road and lines.

Input shape of the model is 320 x 320, and we resize the input video to the required shape.

## Demo

![Image example](media/yolop.gif)

Example shows input video with overlay of lane and line segmentation and vehicle detections. Example video is taken from [YOLOP repository](https://github.com/hustvl/YOLOP/tree/main/inference/videos).

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

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

This will run the YOLOP experiment with the default device, and camera input.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the YOLOP experiment with the default device and the video file.

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

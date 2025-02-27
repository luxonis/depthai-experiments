# MJPEG Streaming

This experiments demonstrate how to stream frames via HTTP server using MJPEG stream. It performs object detection using [YOLOv6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model.

## Demo

![example](media/example.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode). To see the streamed frames, open [http://localhost:8083](http://localhost:8083).

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

This will run the MJPEG Streaming experiment with the default device and camera input.

```
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the MJPEG Streaming experiment with the default device and the video file.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

# TODO: add instructions for standalone mode once oakctl supports CLI arguments

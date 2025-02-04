# Lossless Zooming

This experiment demonstrates how to perform lossless zooming on the device. It will zoom into the first face it detects. It will crop 1080p frames, centered around the face. The experiment uses [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915) NN model to detect faces.

## Demo

![example](media/example.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your computer as host (`PERIPHERAL` mode).

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

This will run the lossless zooming experiment with the default device and camera input.

```
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the lossless zooming experiment with the default device and the video file.

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

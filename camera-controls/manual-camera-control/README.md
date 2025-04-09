# Manual Camera Control

This experiment demonstrates how to manually control different camera parameters. Use keyboard to modify different settings.

## Demo

![example](media/example.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

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
                    FPS limit for the model runtime. (default: 30)
```

#### Examples

```bash
python3 main.py
```

This will run the manual camera control experiment with the default device and camera input.

### Standalone Mode

Running the example in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/), app runs entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.

## Keyboard Controls

| Key | Description |
| -------- | ------------------------------------ |
| `c` | Capture an image |
| `e` | Autoexposure |
| `t` | Trigger autofocus |
| `f` | Autofocus (continuous) |
| `w` | Auto white balance lock (true/false) |
| `r` | Auto exposure lock (true/false) |
| `+`, `-` | Increase/decrease selected control |

The following controls can be selected and modified with `+` and `-` keys:

| Key | Description |
| --- | -------------------------- |
| `1` | Manual exposure time |
| `2` | Manual sensitivity ISO |
| `3` | Auto white balance mode |
| `4` | Auto exposure compensation |
| `5` | Anti-banding/flicker mode |
| `6` | Effect mode |
| `7` | Brightness |
| `8` | Contrast |
| `9` | Saturation |
| `0` | Sharpness |
| `o` | Manual white balance |
| `p` | Manual focus |
| `[` | Luma denoise |
| `]` | Chroma denoise |

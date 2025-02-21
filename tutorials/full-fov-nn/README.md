# Full FOV NN inferencing

See [Resolution Techniques for NNs](https://docs.luxonis.com/software/depthai/resolution-techniques) for more information.

This experiment demonstrates how to run NN inferencing on full FOV frames. It uses a video stream with a different aspect ratio than the NN input. [YOLOv6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) is used for object detection.

There are 3 options, how to match the NN input aspect ration:

1. Crop the original frame before inferencing and lose some FOV
1. Apply letterboxing to the frame to get the correct aspect ratio and lose some accuracy
1. Stretch the frame to the correct aspect ratio of the NN and lose some accuracy

## Demo

### Cropping

![cropping example](media/crop_example.jpg)

### Letterboxing

![letterboxing example](media/letterbox_example.jpg)

### Stretching

![stretching example](media/stretch_example.jpg)

## Installation

```bash
pip install -r requirements.txt
```

## Scripts

This experiment contains 4 different scripts.

### `main.py`

This is the main script that runs the experiment and lets you choose the resize mode during runtime by using the following keybinds:

| Key | Mode         |
| --- | ------------ |
| a   | Letterboxing |
| s   | Crop         |
| d   | Stretch      |

### `letterboxing.py`, `cropping.py` and `stretch.py`

These scripts run only in the corresponding mode, which cannot be toggled during runtime.

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters (these are available for all scripts):

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

This will run the Full FOV NN inferencing experiment with the default device and camera input.

```bash
python3 cropping.py -fps 10
```

This will run the Full FOV NN inferencing experiment using cropping resize mode with the default device at 10 FPS.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

# TODO: add instructions for standalone mode once oakctl supports CLI arguments

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

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).

Install required packages by running:

```bash
pip install -r requirements.txt
```

## Scripts

This experiment contains 4 different scripts.

### `main.py`

This is the main script that runs the experiment and lets you choose the resize mode during runtime by using the following keybinds:

| Key | Mode |
| --- | ------------ |
| a | Letterboxing |
| s | Crop |
| d | Stretch |

### `letterboxing.py`, `cropping.py` and `stretch.py`

These scripts run only in the corresponding mode, which cannot be toggled during runtime.

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your computer as host (`PERIPHERAL` mode). `STANDALONE` mode is only supported on RVC4.

All scripts accept the following arguments:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

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

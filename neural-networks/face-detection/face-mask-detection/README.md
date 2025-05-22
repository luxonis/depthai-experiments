# Face Mask Detection

This experiment demonstrates how to build a single-stage DepthAI pipeline for face mask detection.
It recognizes whether the human faces detected on the frame are (not) wearing face masks.
The pipeline consists of the [PPE Detection](https://zoo-rvc4.luxonis.com/luxonis/ppe-detection/fd8699bf-3819-4134-9374-3735b9660d3c) model trained to recognize person protective equipment (PPE).

## Demo

[![face mask detection](media/face_mask_detection.gif)](media/face_mask_detection.gif)

<sup>[Source](https://www.pexels.com/video/woman-art-iphone-smartphone-3960181/)</sup>

## Usage

Running this example requires a **Luxonis device** connected to your computer. Refer to the [documentation](https://stg.docs.luxonis.com/software/) to setup your device if you haven't done it already.

You can run the experiment fully on device ([`STANDALONE` mode](#standalone-mode-rvc4-only)) or using your your computer as host ([`PERIPHERAL` mode](#peripheral-mode)).

Here is a list of all available parameters:

```
-m MODEL, --model MODEL
                      HubAI model reference. (default: 'luxonis/ppe-detection:640x640')
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                      FPS limit for the model runtime. (default: 5 for RVC2 and 30 for RVC4)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

## Peripheral Mode

### Installation

You need to first prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

### Examples

```bash
python3 main.py
```

This will run the experiment with default arguments.

## Standalone Mode (RVC4 only)

Running the example in the standalone mode, app runs entirely on the device.
To run the example in this mode, first install the `oakctl` tool using the installation instructions [here](https://stg.docs.luxonis.com/software/oak-apps/oakctl).

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file (refer [here](https://stg.docs.luxonis.com/software/oak-apps/configuration/) for more information about this configuration file).

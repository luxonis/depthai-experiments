# Gaze estimation

**:exclamation: This experiment works only on RVC4 devices:exclamation:**
This example demonstrates how to run a 3 stage pipeline and multi input models.

[![Gaze Example Demo](https://github.com/luxonis/depthai-experiments/assets/18037362/6c7688e5-30bc-4bed-8455-8b8e9899c5b0)](https://tinyurl.com/5h3dycc5)

## Description

A gaze estimation example is built to showcase the 3 stage pipeline. The pipeline is composed of the following three models:

1. [SCRFD face detection model](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page) is used to detect the faces and their keypoints. Based on this models outputs, the entire face and the eyes are cropped from the original frame.
1. The cropped face is used as input to the [Head pose model](https://hub.luxonis.com/ai/models/068ac18a-de71-4a6e-9f0f-42776c0ef980?view=page) which returns the 3D vector the heads position.
1. The cropped eyes and the 3D pose vector are fed into [ADAS gaze estimation model](https://hub.luxonis.com/ai/models/b174ff1b-740b-4016-b8d5-b9488dbdd657?view=page) to compute the final gaze of the person.

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

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
                      FPS limit for the model runtime. (default: 3 for RVC2 and 30 for RVC4)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py \
    --device <DEVICE IP OR MXID>
```

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

# Animal detection and pose estimation

This experiment demonstrates how to build a 2-stage DepthAI pipeline for detecting animals and estimating their poses. The pipeline consists of [YOLOv6](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034?view=page) object detector and [SuperAnimal](https://hub.luxonis.com/ai/models/894cf1a2-23fb-4c96-8944-a0d1be38a7c7?view=page) pose estimation model. The experiment works on both RVC2 and RVC4. For realtime application you will need to use OAK4 cameras.

## Demo

[![Animal detection and pose estimation](media/cow-walking.gif)](media/cow-walking.gif)

<sup>[Source](https://www.youtube.com/shorts/LGof_auMHuc)</sup>

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

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
                    FPS limit for the model runtime. (default: 5 for RVC2 and 20 for RVC4)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the animal detection and pose estimation experiment with the default device and camera input.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the animal detection and pose estimation experiment with the default device and the video file.

```bash
python3 main.py --device <DEVICE IP OR MXID>
```

This will run the animal detection and pose estimation experiment with the specified device and camera input.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
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

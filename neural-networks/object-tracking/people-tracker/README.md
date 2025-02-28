# People tracker

This experiment counts how many people went up / down / left / right in the video stream, allowing you to
receive an information about eg. how many people went into a room or went through a corridor. It uses [SCRFD Person detection](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c)

## Demo

![example](media/example.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-t THRESHOLD, --threshold THRESHOLD
                    Minimum distance the person has to move (across the x/y axis) to be considered a real movement. (default: 0.25)
```

### Peripheral Mode

```bash
python3 main.py
```

This will run the People Tracker experiment with the default device and camera input.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the People Tracker experiment with the default device and the video file.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

# TODO: add instructions for standalone mode once oakctl supports CLI arguments

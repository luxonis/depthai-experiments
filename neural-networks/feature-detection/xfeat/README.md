# XFeat: Accelerated Features for Lightweight Image Matching

## Overview

This example demonstrates how to use the XFeat model from Luxonis HubAI with the DepthAI platform. XFeat has compact descriptors (64D) and simple architecture components that facilitate deployment on embedded devices. Performance is comparable to known deep local features such as SuperPoint while being significantly faster and more lightweight.

The model used in the example is available in our HubAI Model ZOO [here](https://hub.luxonis.com/ai/models/6c2790a1-bf68-4e89-a4b3-5c9ae68183b5?view=page).

We offer two modes of operation: `mono` and `stereo`. In `mono` mode we use a single camera as an input and match the frames to the reference image. In `stereo` mode we use two cameras to match the frames to each other.
In `mono` mode we visualize the matches between the frames and the reference frame which can be set by pressing `s` key in the visualizer. In `stereo` mode we visualize the matches between the frames from the left and right camera.

## Demo
    
![XFeat Mono Demo on OAK](media/xfeat_demo.gif)

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --model <MODEL> --device <DEVICE> --fps_limit <FPS_LIMIT>
```


- `<MODEL>`: HubAI Model Reference from Luxonis HubAI. Default: `luxonis/xfeat:mono-240x320`.
- `<DEVICE>`: Device IP or ID. Default: ``.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

If you use model with mono mode (e.g. ``luxonis/xfeat:mono-240x320``), you can set reference frame by pressing `s` key in the visualizer.

**NOTE**: Stereo mode will run only with 2-camera devices.

#### Examples

```bash
python3 main.py
```

This will run the XFeat model in mono mode with the default model and device. Default model: `luxonis/xfeat:mono-240x320`.

```
python3 main.py --model luxonis/xfeat:mono-480x640
```

This will run the XFeat model in mono mode with the `luxonis/xfeat:mono-480x640` model. This model is more accurate but slower than the default model.

```
python3 main.py --model luxonis/xfeat:stereo-240x320
```

This will run the XFeat model in stereo mode with the `luxonis/xfeat:stereo-240x320` model. The model will match the frames from two cameras (e.g. left and right camera).

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:
```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```
and run the example using the `run_standalone.py` script:
```bash
python3 run_standalone.py \
    --model <Model> \
    --device <Device> \
    --fps_limit <FPS>
```

The arguments are the same as in the Peripheral mode.

#### Example
```bash
python3 run_standalone.py \
    --model xfeat:stereo-240x320 \
```
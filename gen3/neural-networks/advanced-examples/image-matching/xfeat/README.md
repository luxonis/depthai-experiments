# XFeat: Accelerated Features for Lightweight Image Matching

## Overview

This example demonstrates how to use the XFeat model from Luxonis HubAI with the DepthAI platform. XFeat has compact descriptors (64D) and simple architecture components that facilitate deployment on embedded devices. Performance is comparable to known deep local features such as SuperPoint while being significantly faster and more lightweight.

We offer two modes of operation: `mono` and `stereo`. in `mono` mode we use a single camera as an input and match the frames to the reference image. In `stereo` mode we use two cameras to match the frames to each other.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py --model <MODEL> --device <DEVICE> --fps_limit <FPS_LIMIT>
```


- `<MODEL>`: Model slug from Luxonis HubAI. Default: `luxonis/xfeat:mono-240x320`.
- `<DEVICE>`: Device IP or ID. Default: ``.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

If you use model with mono mode (e.g. ``luxonis/xfeat:mono-240x320``), you can set reference frame by pressing `s` key in the visualizer.

**NOTE**: Stereo mode will run only with 2-camera devices.

## Example

```
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
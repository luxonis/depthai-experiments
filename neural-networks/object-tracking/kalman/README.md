# Kalman Filter

This example performs filtering of 2D bounding boxes and spatial coordinates of tracked objects using the Kalman filter.

The Kalman filter is used to obtain better estimates of objects' locations by combining measurements and past estimations. Even when measurements are noisy this filter performs quite well.

Here is a short explanation of the Kalman filter: https://www.youtube.com/watch?v=s_9InuQAx-g.

It uses our [YOLOv6 nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model for detection.

> **Note:** This example requires a device with at least 3 cameras (color, left and right) since it utilizes the `StereoDepth` node.

> **Note:** This example currently only works on RVC2 devices becuase dai.ObjectTracker node is not supported on RVC4.

## Demo

![video](https://user-images.githubusercontent.com/69462196/197813200-236e950e-3dda-403f-b5cd-8d11f0e86124.gif)

In the demo red represents filtered data and blue unfiltered.

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
                    FPS limit for the model runtime. (default: 20)
```

#### Examples

```bash
python3 main.py
```

This will run the Kalman filter experiment with the default device and camera input.

```bash
python3 main.py --fps_limit 10
```

This will run the Kalman filter experiment with the default device and 10 FPS limit.

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

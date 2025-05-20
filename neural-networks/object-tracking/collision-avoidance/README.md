# Collision avoidance

This experiment's goal is to detect the objects moving towards the camera and alert the user if it can be dangerous pass. We use our [YOLOv6 nano](https://hub.luxonis.com/ai/models/face58c4-45ab-42a0-bafc-19f9fee8a034) model for detecting the desired objects. By default, it detects the class `person` but it can be easily changed to any other class like `car`, `bicycle`, etc. (just modify the variable in the `main.py` file). The app also tracks the objects and estimates their position in 3D space using our depth cameras (you need to have depth-enabled cameras connected). Dangerous pass is detected when the object is moving towards the camera, this is done by checking the direction of the object's trajectory.

You can see the visualization of the object's trajectory in the `Direction` topic. We also visualize the bird's eye view of the scene.

> **Note:** This example requires a device with at least 3 cameras (color, left and right) since it utilizes the `StereoDepth` node.

> **Note:** This example currently only works on RVC2 devices becuase dai.ObjectTracker node is not supported on RVC4.

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

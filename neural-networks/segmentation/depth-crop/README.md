# Depth cropping with Deeplabv3+

This example shows how to run the [Deeplabv3+](https://hub.luxonis.com/ai/models/1189a661-fd0a-44fd-bc9e-64b94d60cb49?view=page) model from our HubAI along with the [StereoDepth](https://rvc4.docs.luxonis.com/software/depthai-components/nodes/stereo_depth/) node and crop the depth image based on the models output.

> **Note:** This example requires a device with at least 3 cameras (color, left and right) since it utilizes the `StereoDepth` node.

## Demo

![Deeplabv3 Depth GIF](https://user-images.githubusercontent.com/59799831/132396685-c494f21b-8101-4be4-a787-dd382ae6b470.gif)

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
                    FPS limit for the model runtime. (default: 10 for RVC2 and 25 for RVC4)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

```bash
python3 main.py
```

This will run the experiment with the default device and camera input.

```bash
python3 main.py --device <DEVICE_ID>
```

This will run the experiment with the specified device.

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

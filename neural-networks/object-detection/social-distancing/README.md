# Social distancing

This experiment demonstates how we can use DepthAI to monitor social distancing. It uses our depth-enabled OAK camera and on-device AI processing. For detecting people we use [SCRFD Person detection model](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c?view=page) from HubAI. We merge the detections with depth information to get the 3D position of each person. We then calculate the distance between each pair of people and if the distance is less than a threshold, we display a warning.

Below you can see 3 people in a scene. If they get closer than the threshold of 2 meters, the application will display `Too Close` and the distance between them.

> **Note:** This example requires a device with at least 3 cameras (color, left and right) since it utilizes the `StereoDepth` node.

## Demo

[![COVID-19 Social Distancing with DepthAI](https://user-images.githubusercontent.com/5244214/90741333-73f89500-e2cf-11ea-919b-b1f47dc55c4a.gif)](https://www.youtube.com/watch?v=-Ut9TemGZ8I "DepthAI Social Distancing Proof of Concept")

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
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the Social Distancing experiment with the default device.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.

## Deep Dive on How it Works

![Social Distancing explanation](https://user-images.githubusercontent.com/32992551/101372410-19c51500-3869-11eb-8af4-f9b4e81a6f78.png)

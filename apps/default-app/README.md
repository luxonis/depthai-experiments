# Default Application 
This application performs real-time object detection using a YOLOv6-nano model and stereo depth estimation (if the device has stereo cameras). It streams raw video, H.264/MJPEG encoded video, object detection results, and a colorized depth map to a remote visualizer for monitoring and analysis.

## Demo
![Demo](./media/demo.gif)

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

You can run the experiment using your your computer as host.

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-api API_KEY, --api_key API_KEY
                      HubAI API key to access private model. (default: )
```

#### Examples

```bash
python3 main.py
```

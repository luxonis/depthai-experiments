# Calculate spatial coordinates on the host

This example shows how you can use multiple DepthAI's on a single host. The demo will find all devices connected to the host and display an RGB preview from each of them.

## Demo

[![Multiple devices per host](https://user-images.githubusercontent.com/18037362/113307040-01d83c00-9305-11eb-9a42-c69c72a5dba5.gif)](https://www.youtube.com/watch?v=N1IY2CfhmEc "Multiple devices per host")

Just two OAK cameras looking at each other.

Program will also print USB speed, and connected cameras for each connected device before starting the pipeline. Output example for having connected OAK-D S2, OAK-D PRO

```
Found 2 DepthAI devices to configure.

Attempting to connect to device: 19443010F1E61F1300...
=== Successfully connected to device: 19443010F1E61F1300
    >>> Cameras: ['CAM_A', 'CAM_B', 'CAM_C']
    >>> USB speed: SUPER
    Pipeline created for device: 19443010F1E61F1300
    Pipeline for 19443010F1E61F1300 configured. Ready to be started.

Attempting to connect to device: 14442C1011D6C5D600...
=== Successfully connected to device: 14442C1011D6C5D600
    >>> Cameras: ['CAM_A', 'CAM_B', 'CAM_C']
    >>> USB speed: SUPER
    Pipeline created for device: 14442C1011D6C5D600
    Pipeline for 14442C1011D6C5D600 configured. Ready to be started.

```

## Usage

You can run the experiment using your your computer as host, ie. in `PERIPHERAL` mode.

Here is a list of all available parameters:

```
--include-ip          Also include IP-only cameras (e.g. OAK-4) in the device list
--max-devices MAX_DEVICES
                        Limit the total number of devices to this count
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app.
You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

#### Examples

```bash
python main.py
```

This will run the demo using only internal DepthAI cameras.

```bash
python main.py --include-ip
```

This will also discover and use any TCP/IP cameras on the network.

```bash
python main.py --max-devices 3
```

This will stop after configuring the first 3 devices.

```bash
python main.py --include-ip --max-devices 3
```

This will include IP cameras and then only use the first 3 discovered devices.

#### Object detection on multiple devices

Script [multi-device-mobilenet.py](multi-device-mobilenet.py) will run `mobilenet-ssd` single shot object detector on all devices and display detections on frames.
If you would want to display detections on high-res frames (not 300x300), check [tutorial here](https://docs.luxonis.com/projects/api/en/latest/tutorials/dispaying_detections/).

![Demo image](https://user-images.githubusercontent.com/18037362/146223605-e4fd0fb3-7cf9-40a0-87e0-73d63a46eb2d.png)

Run this script with

```bash
python3 multi-device-mobilenet.py [--include-ip] [--max-devices N]
```

#### MJPEG decoding (threaded) on multiple devices

The [multi-device-mjpeg-decoding.py](multi-device-mjpeg-decoding.py) script will connect to multiple devices and stream encoded 4K JPEG frames from all devices to the host. On host computer we run a thread for each device, boot the device, get frames,
decode them, and put them in a queue which the main thread reads and displays the frame on the main thread.

Run this script with

```bash
python3 multi-device-mjpeg-decoding.py [--include-ip] [--max-devices N]
```

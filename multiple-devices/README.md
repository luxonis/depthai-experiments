# Multiple devices per host

This example shows how you can use multiple DepthAI's on a single host. The demo will find all devices connected to the host and display an RGB preview from each of them. Preview size will depend on the device type - if device has mono cameras, the rgb preview will be `600x300`, otherwise it will be `300x300`. In other words,  pipeline that will get uploaded to the device will depend on the device model (OAK-1/OAK-D).

## Demo

[![Multiple devices per host](https://user-images.githubusercontent.com/18037362/113307040-01d83c00-9305-11eb-9a42-c69c72a5dba5.gif)](https://www.youtube.com/watch?v=N1IY2CfhmEc "Multiple devices per host")

Just two OAK cameras looking at each other.

Program will also print USB speed, model type and connected cameras for each connected device before starting the pipeline. Output example for having connected an OAK-1 on USB2, OAK-D on USB3, OAK-1-POE and OAK-D-POE:

```
Found 4 devices
=== Connected to 14442C10016B5ED700
   >>> MXID: 14442C10016B5ED700
   >>> Cameras: RGB LEFT RIGHT
   >>> USB speed: SUPER
   >>> Loading pipeline for: OAK-D
=== Connected to 14442C10D197AACE00
   >>> MXID: 14442C10D197AACE00
   >>> Cameras: RGB
   >>> USB speed: HIGH
   >>> Loading pipeline for: OAK-1
[192.168.1.23] [29.087] [system] [warning] Calibration Data on device is empty
=== Connected to 192.168.1.23
   >>> MXID: 14442C1031A3A7D200
   >>> Cameras: RGB LEFT RIGHT
   >>> USB speed: UNKNOWN
   >>> Loading pipeline for: OAK-D-POE
=== Connected to 192.168.1.27
   >>> MXID: 14442C1041D0A7D200
   >>> Cameras: RGB
   >>> USB speed: UNKNOWN
   >>> Loading pipeline for: OAK-1-POE
```

## Object detection on multiple devices

Script `multi-device-mobilenet.py` will run `mobilenet-ssd` single shot object detector on all devices and display detections on frames.
If you would want to display detections on high-res frames (not 300x300), check [tutorial here](https://docs.luxonis.com/projects/api/en/latest/tutorials/dispaying_detections/).

![Demo image](https://user-images.githubusercontent.com/18037362/146223605-e4fd0fb3-7cf9-40a0-87e0-73d63a46eb2d.png)

## MJPEG decoding (threaded) on multiple devices

The [multi-device-mjpeg-decoding.py](multi-device-mjpeg-decoding.py) script will connect to multiple devices and stream encoded 4K JPEG frames from all devices to the host. On host computer we run a thread for each device, boot the device, get frames,
decode them, and put them in a queue which the main thread reads and displays the frame on the main thread.
## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

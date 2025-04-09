# Calculate spatial coordinates on the host

This example shows how to calcualte spatial coordinates of a ROI on the host and gets depth frames from the device. Other option is to use [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/components/nodes/spatial_location_calculator/) to calcualte spatial coordinates on the device.

If you already have depth frames and ROI (Region-Of-Interest, eg. bounding box of an object) / POI (Point-Of-Interest, eg. feature/key-point) on the host, it might be easier to just calculate the spatial coordiantes of that region/point on the host, instead of sending depth/ROI back
to the device.

> **Note**: Using single points / tiny ROIs (eg. 3x3) should be avoided, as depth frame can be noisy, so you should use **at least 10x10 depth pixels
> \<\<\<\<\<\<\< HEAD
> ROI**. Also note that to maximize spatial coordinates accuracy, you should define min and max threshold accurately.
> \=======
> ROI\*\*. Also note that to maximize spatial coordinates accuracy, you should define min and max threshold accurately.
>
> > > > > > > 9f415997 (refactoring spatial calculation on host)

## Demo

![Demo](https://user-images.githubusercontent.com/18037362/146296930-9e7071f5-33b9-45f9-af21-cace7ffffc0f.gif)

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

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
```

### Peripheral Mode

# \<\<\<\<\<\<\< HEAD Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. You can run the example with the following command:

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. You can run the example with the following command:

> > > > > > > 9f415997 (refactoring spatial calculation on host)

```bash
python3 main.py
```

### Standalone Mode

Running the example in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/), app runs entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

# This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file. \<\<\<\<\<\<\< HEAD

> > > > > > > 9f415997 (refactoring spatial calculation on host)

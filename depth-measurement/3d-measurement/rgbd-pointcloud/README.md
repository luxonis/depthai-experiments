## RGB-D projection Demo

This example shows how to align depth to rgb camera frame and project depth map into 3D pointcloud. You can also choose to skip rgb-depth alignment and colorize the pointcloud with right mono frame.

## Demo
![ezgif-72a39b6d489bd8](https://github.com/user-attachments/assets/2a742929-f080-4a1f-8db5-da356b695876)
![ezgif-7736189d82e875](https://github.com/user-attachments/assets/2cb05ac6-1dca-421b-88a9-d86c05c6e4e1)

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
python3 main.py
```

```
optional arguments:
    -d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
    -m, --mono            use mono frame for pointcloud coloring instead of color frame
```

### Peripheral Mode

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),

You can simply install them by running:

```bash
pip install -r requirements.txt
```

You can run the app using the following command:

```bash
python3 main.py
```

```bash
python3 main.py -m
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

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.

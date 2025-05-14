# Stereo Runtime Configuration

This example showcases, how to reconfigure the [StereoDepth](https://rvc4.docs.luxonis.com/software/depthai-components/nodes/stereo_depth/) node during runtime. It shows a preview of the depth map and allows you to change the stereo depth settings using your keyboard.

## Demo

![example](media/example.png)

## Installation

Running this example requires a stereo depth capable **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).

Install required packages by running:

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
                    FPS limit. (default: 20)
```

To change the stereo depth settings, use the following keys:

| Key | Description                     |
| --- | ------------------------------- |
| `k` | Change median filtering mode    |
| `l` | Enable/disable left-right check |
| `.` | Increase confidence threshold   |
| `,` | Decrease confidence threshold   |

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the example with the default device and camera input.

```bash
python3 main.py -fps 10
```

This will run the example with the default device and camera input at 10 FPS.

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

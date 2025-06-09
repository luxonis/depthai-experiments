# Speech Recognition

This example demonstrates how to run
[Whisper Tiny EN Network](https://zoo-rvc4.luxonis.com/luxonis/whisper-tiny-en/0aaf1b77-761b-44d6-893c-c473ca463186) on DepthAI with OAK4 devices.

> **Note:** This example only works on RVC4.

## Demo

## Usage

Running this example requires a **Luxonis device** connected to your computer. Refer to the [documentation](https://stg.docs.luxonis.com/software-v3/) to setup your device if you haven't done it already.

You can run the example fully on device ([`STANDALONE` mode](#standalone-mode-rvc4-only)) or using your computer as host ([`PERIPHERAL` mode](#peripheral-mode)).

Here is a list of all available parameters:

```
--device_ip DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
--audio_file
                    The audio file that will be used in the example.
```

## Peripheral Mode

### Installation

You need to first prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

You will need to install the `ffmpeg` tool that is needed for audio pre-processing.

```bash
sudo apt-get install ffmpeg
```

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

### Examples

```bash
python3 main.py --device_ip <device_ip> --audio_file <audio_file>
```

> \[!WARNING\]
> The `--device_ip` and `--audio_file` arguments are mandatory. The `device_ip` is the IP address of the device you are connecting to. The `audio_file` is the path to the audio file you want to use for the inference.

## Standalone Mode (RVC4 only)

Running the example in the standalone mode, app runs entirely on the device.
To run the example in this mode, first install the `oakctl` tool using the installation instructions [here](https://stg.docs.luxonis.com/software-v3/oak-apps/oakctl).

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

This will run the example with default argument values. If you want to change these values you need to edit the `oakapp.toml` file (refer [here](https://stg.docs.luxonis.com/software-v3/oak-apps/configuration/) for more information about this configuration file).

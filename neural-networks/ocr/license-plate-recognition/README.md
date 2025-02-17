## License Plate Recognition

This example demonstrates how to run 3 stage inference on DepthAI
First, a vehicle is detected on the image, the cropped image is then fed into a license plate detection model. The cropped license plate is sent to a text recognition (OCR) network,
which tries to decode the license plates texts.

**:exclamation: Due to the high computational cost, this example only works on OAK4 devices. :exclamation:**

Take a look at [How to Train and Deploy a License Plate Detector to the Luxonis OAK](https://blog.roboflow.com/oak-deploy-license-plate/) tutorial for training a custom detector using the Roboflow platform.

## Demo

![Detection Output](media/lpr.gif)

<sup>[Source](https://www.pexels.com/video/speeding-multicolored-cars-trucks-and-suv-motor-vehicles-exit-a-dark-new-york-city-underground-tunnel-which-is-wrapped-in-the-lush-green-embrace-of-trees-and-bushes-17108719/)</sup>

## Instalation

Running this example requires a **Luxonis OAK4 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Example

```bash
python3 main.py \ 
        -d <device ip / mxid>
```

This will run the experiment on the specifid device.

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

# Gen3 Stereo on host

> Your device needs to have calibration stored to work with this example

This experiment demonstrates how [stereo pipeline works](https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/#internal-block-diagram-of-stereodepth-node) on the OAK device (using depthai). It rectifies mono frames (receives from the OAK camera) and then uses `cv2.StereoSGBM` to calculate the disparity on the host. It also colorizes the disparity and shows it to the user.

[SSIM score](https://en.wikipedia.org/wiki/Structural_similarity) is used to comapre the similarity between the disparity map calculated using `cv2.StereoSGBM` and the disparity map generated on the OAK camera.

> You can play around with the settings for both methods and use the SSIM score to compare them.

## Demo
<img width="1193" alt="Screenshot 2025-04-24 at 11 43 55" src="https://github.com/user-attachments/assets/4eba827b-7515-432d-b89e-c0c993922313" />

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
```

### Peripheral Mode

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

You can run the app using the following command:

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

This will run the experiment with default argument values. If you want to change these values you need to edit the `oakapp.toml` file.

## Host-Side WLS Filtering

This gives an example of doing host-side WLS filtering using the `rectified_right` and `depth` stream from DepthAI Gen3 API.

## Demo

<img width="1239" alt="Screenshot 2025-05-02 at 13 46 15" src="https://github.com/user-attachments/assets/57f2f66e-a0a3-44b3-bb88-a2ae4a8c6e93" />

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
```

### Controls

Use the following keyboard controls in the visualizer to adjust WLS filtering parameters:

| Key | Action            |
| --- | ----------------- |
| `l` | Decrease lambda   |
| `L` | Increase lambda   |
| `s` | Decrease sigma    |
| `S` | Increase sigma    |
| `q` | Quit the pipeline |

- **Lambda** controls the amount of smoothing (higher = smoother).
- **Sigma** controls the edge sensitivity (higher = more edge-aware).

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

You can run the example with the following command:

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

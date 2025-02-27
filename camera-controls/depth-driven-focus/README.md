# Depth Driven Focus

This experiment demonstrates how to continuously focus on the first detected face. It will determine the distance to the face and adjust the lens position accordingly. The experiment requires a device with an auto-focus color camera and a stereo camera pair to function properly. The experiment uses [YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915) NN model to detect faces.

## Demo

TODO

## Installation

You need to prepare a Python 3.10 environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --fps_limit <FPS_LIMIT>
```

- `<DEVICE>`: Device IP or ID. Default: \`\`.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

#### Examples

```bash
python3 main.py
```

This will run the depth driven focus experiment with the default device and camera input.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:

```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```

and run the example using the `run_standalone.py` script:

```bash
python3 run_standalone.py \
    --device <DEVICE IP> \
    --fps_limit <FPS>
```

The arguments are the same as in the Peripheral mode.

#### Example

```bash
python3 run_standalone.py \
    --fps_limit 20 \
```

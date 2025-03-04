# Custom models

This experiment demonstrates, how to create custom models with PyTorch/Kornia, convert them and run them with DepthAI. For more information see [Conversion](https://rvc4.docs.luxonis.com/software/ai-inference/conversion/) section in the documentation and [README.md](generate_model/README.md) file in the `generate_model/` folder.

`blur.py`, `concat.py`, `diff.py`, `edge.py` and `main.py` are scripts that run created custom models. `generate_model/` folder contains scripts that create these custom models (frame blurring, frame concatenation, frame difference and edge detection).

## Installation

### Running the models

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

### Generating the models

To generate the models, you need to install packages specified in `generate_model/requirements.txt` file:

```bash
pip install -r generate_model/requirements.txt
```

For more information see [README.md](generate_model/README.md) file in the `generate_model/` folder.

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect
                    to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example:

#### Examples

```bash
python3 main.py
```

This will run all models at once on your camera input.

```bash
python3 blur.py
```

This will run the blurring model on your camera input.

```bash
python3 concat.py -fps 10
```

This will run the concatenation model on your camera input with FPS set to 10.

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

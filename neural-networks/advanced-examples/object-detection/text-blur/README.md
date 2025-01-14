## Text blurring on DepthAI

This example shows an implementation of [Text Detection](https://hub.luxonis.com/ai/models/131d855c-60b1-4634-a14d-1269bb35dcd2?view=page) on DepthAI in the Gen3 API system with additional text blurring.

![Image example](imgs/output.gif)

# Instalation
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:
```bash
pip install -r requirements.txt
```

Install requirements:
```
python3 -m pip install -r requirements.txt
```

# Usage
The inference is ran using a simple CLI call:
```bash
python3 main.py \
    --device ... \
    --media ...
```

The relevant arguments:
- **--device** [OPTIONAL]: DeviceID or IP of the camera to connect to.
By default, the first locally available device is used;
- **--media** [OPTIONAL]: Path to the media file to be used as input. 
Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results in a web browser using **Depthai Visualizer**.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:
```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```
The app can then be run with:
```bash
oakctl connect <device-ip>
oakctl app run .
```

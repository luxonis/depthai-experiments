## Text blurring on DepthAI

This example shows an implementation of [YuNet face detection](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) on DepthAI v3 with additional blurring.

![Image example](visualizations/output.gif)

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

- **--device** \[OPTIONAL\]: DeviceID or IP of the camera to connect to.
  By default, the first locally available device is used;
- **--media** \[OPTIONAL\]: Path to the media file to be used as input.
  Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
  By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results in a web browser using **Depthai Visualizer**.

## Example

To run the example you can simply run the following command:

```bash
python3 main.py \ 
        -d <<device ip / mxid>>
```

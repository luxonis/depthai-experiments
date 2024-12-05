# Overview
We provide here an example for running inference with a **single model** on a **single-image input**.
The example is generic and can be used for various single-image input models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models).

# Instalation
Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and how to set them up in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:
- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

## DepthAI
As **DepthAI v3** is not officially released yet, you need to install it from the artifacts:
```bash
python3 -m pip install -U --prefer-binary --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local "depthai==3.0.0-alpha.6.dev0+1eb20bd1b5b598282271a790713b9eb65b7546a7"
```

## DepthAI Nodes
As **DepthAI Nodes** is still work-in-progress, we suggest installing the latest version from the **GitHub**:
```bash
git clone git@github.com:luxonis/depthai-nodes.git
cd depthai-nodes
pip install -e .
```

# Usage
The example is run using a simple CLI call:
```bash
python3 main.py \
    --model_slug ... \
    --fps_limit ... \
    --device ... \
    --media ...
```

The relevant arguments:
- **--model_slug**: A unique HubAI identifier of the model;
- **--fps_limit** [OPTIONAL]: The upper limit for inference runtime in FPS (frames per second). 
By default, this is set to 30.
If using OAK-D Lite, make sure to set it under 28.5;
- **--device** [OPTIONAL]: DeviceID or IP of the camera to connect to.
By default, the first locally available device is used;
- **--media** [OPTIONAL]: Path to the media file (image or video) to be used as input.
By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media file, and display the results by **DepthAI visualizer**
The latter runs in the browser at `http://localhost:8082`.
In case of a different client, replace `localhost` with the correct hostname.

## Example
To try it out, let's run a simple YOLOv6 object detection model on your camera input.
```bash
python3 main.py \
    --model_slug luxonis/yolov6-nano:r2-coco-512x288
```

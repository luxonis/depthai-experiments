# Overview
We provide here an example for running inference with a **single model** on a **single-image input** with a **single-head output**.
The example is generic and can be used for various single-image input models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models).

# Instalation
Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
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
The inference is ran using a simple CLI call:
```bash
python3 main.py \
    --model_slug ... \
    --device ... \
    --annotation_mode ... \
    --fps_limit ... \
    --media ...
```

The relevant arguments:
- **--model_slug**: A unique HubAI identifier of the model;
- **--device** [OPTIONAL]: DeviceID or IP of the camera to connect to.
By default, the first locally available device is used;
- **--annotation_mode** [OPTIONAL]: Annotation mode. Set to 'segmentation_overlay' to overlay segmentation masks over the model inputs. Leave empty to use the default visualization.
- **--fps_limit** [OPTIONAL]: The upper limit for camera captures in frames per second (FPS).
The limit is not used when infering on media.
By default, the FPS is not limited.
If using OAK-D Lite, make sure to set it under 28.5;
- **--media** [OPTIONAL]: Path to the media file to be used as input. 
Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results by **DepthAI visualizer**
The latter runs in the browser at `http://localhost:8082`.
In case of a different client, replace `localhost` with the correct hostname.

## Example
To try it out, let's run a simple YOLOv6 object detection model on your camera input.
```bash
python3 main.py \
    --model_slug luxonis/yolov6-nano:r2-coco-512x288
```
# Overview
We provide here an example for running inference with a **single model** on a **single-image input** with a **single-head output**.
The example is generic and can be used for various single-image input models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models).

# Instalation
Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:
- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:
```bash
pip install -r requirements.txt
```

# Usage

## Peripheral Mode
Running the example in the **peripheral mode** offloads some tasks to a host computer.
The example can be ran in this mode as:
```bash
python3 main.py \
    --model <Model> \
    --device <Device> \
    --annotation_mode <Mode> \
    --fps_limit <FPS> \
    --media <Media>
```

Relevant arguments:
- `<Model>`: A unique HubAI identifier of the model;
- `<Device>` [OPTIONAL]: DeviceID or IP of the camera to connect to.
By default, the first locally available device is used;
- `<Mode>` [OPTIONAL]: Annotation mode. Set to 'segmentation' to overlay segmentation masks over the model inputs, or 'segmentation_with_annotation' to visualize the additional annotations. Leave empty to use the default visualization.
- `<FPS>` [OPTIONAL]: The upper limit for camera captures in frames per second (FPS).
The limit is not used when infering on media.
By default, the FPS is not limited.
If using OAK-D Lite, make sure to set it under 28.5;
- `<Media>` [OPTIONAL]: Path to the media file to be used as input. 
Currently, only video files are supported but we plan to add support for more formats (e.g. images) in the future.
By default, camera input is used;

Running the script downloads the model, creates a DepthAI pipeline, infers on camera input or the provided media, and display the results by **DepthAI visualizer**
The latter runs in the browser at `http://localhost:8082`.
In case of a different client, replace `localhost` with the correct hostname.

### Example
To try it out, let's run a simple YOLOv6 object detection model on your camera input:
```bash
python3 main.py \
    --model luxonis/yolov6-nano:r2-coco-512x288
```

## Standalone Mode
Running the example in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/), app runs entirely on the device.
To run the example in this mode, first install the [oakctl](https://rvc4.docs.luxonis.com/software/tools/oakctl/) command-line tool (enables host-device interaction) as:
```bash
bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"
```
and run the example using the `main_standalone.sh` script:
```bash
bash main_standalone.sh \
    --model <Model> \
    --device <Device> \
    --annotation_mode <Mode> \
    --fps_limit <FPS> \
    --media <Media>
```

The arguments are the same as in the Peripheral mode. 
Note, however, that when specifying a media file to be used as input, it must be located within the `generic-example` folder.
This way it will get automatically transferred to the device.
Therefore, make sure to provide the file's path relative to the `generic-example` folder.

### Example
```bash
bash main_standalone.sh \
    --model luxonis/yolov6-nano:r2-coco-512x288
```
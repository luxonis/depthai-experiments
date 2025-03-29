# On-device Pointcloud NN model

[Click here](https://docs.luxonis.com/en/latest/pages/tutorials/device-pointcloud) for the **tutorial/blog about this demo**.

This demo uses \[custom NN model\](file:///home/erik/Luxonis/depthai-docs-website/build/html/pages/tutorials/creating-custom-nn-models.html#run-your-own-cv-functions-on-device) approach to run custom logic - depth to pointcloud conversion - on the OAK camera itself.

The model was inspired by Kornia's [depth_to_3d](https://kornia.readthedocs.io/en/latest/geometry.depth.html?highlight=depth_to_3d#kornia.geometry.depth.depth_to_3d) function, but due to the slow performance, it was then built with pytorch.

## Demo

![image](https://user-images.githubusercontent.com/18037362/158055419-5c80d524-3478-49e0-b7b8-099b07dd57fa.png)

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

> Running the command above also tries to install open3D which is required for this example.
open3D is not supported by all platforms, but is required for pointcloud visualization. Installing open3D on [Python 3.12 is not yet supported](https://stackoverflow.com/questions/62352767/cant-install-open3d-libraries-errorcould-not-find-a-version-that-satisfies-th).


## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-m MODEL, --model MODEL
                      HubAI model reference. (default: luxonis/yolov6-nano:r2-coco-512x288)
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-ann ANNOTATION_MODE, --annotation_mode ANNOTATION_MODE
                      Annotation mode. Can be either 'segmentation', 'segmentation_with_annotation', or None (default). (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                      FPS limit for the model runtime. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-api API_KEY, --api_key API_KEY
                      HubAI API key to access private model. (default: )
```

**Note:**

If you want to visualize segmentation model output as an overlay you should use `-ann segmentation`. Similarly, if you are using an instance segmentation model and want to visualize its output as overlay you should use `-ann segmentation_with_annotation`.

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run a simple YOLOv6 object detection model (`luxonis/yolov6-nano:r2-coco-512x288`) on your camera input.

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

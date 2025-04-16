# Generic Example

We provide here an example for running inference with a **single model** on a **single-image input** with a **single-head output**.
The example is generic and can be used for various single-image input models from the [HubAI Model ZOO](https://hub.luxonis.com/ai/models).

## Installation

Running this example requires a **Luxonis device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/),
- [DepthAI Nodes](https://pypi.org/project/depthai-nodes/).

You can simply install them by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-m MODEL, --model MODEL
                      HubAI model reference. (default: luxonis/yolov6-nano:r2-coco-512x288)
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                      FPS limit for the model runtime. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-api API_KEY, --api_key API_KEY
                      HubAI API key to access private model. (default: )
-overlay OVERLAY_MODE, --overlay_mode
                      If passed, overlays model output on the input image when the output is an array (e.g., depth maps, segmentation maps). Otherwise, displays outputs separately.
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

```bash
python3 main.py \
    --model luxonis/mediapipe-selfie-segmentation:256x144 \
    --ann segmentation
```

This will run a selfie segmentation model.

```bash
python3 main.py \
    --model luxonis/yolov8-instance-segmentation-nano:coco-512x288 \
    --ann segmentation_with_annotation
```

And this will run an instance segmentation model.

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

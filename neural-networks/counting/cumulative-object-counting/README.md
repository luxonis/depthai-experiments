# Cumulative Object Counting

This example demonstrates how to run an inference pipeline for cummulative object counting using the DepthAI library.
It utilizes an object detection model to detect objects (e.g. `people`) and counts how many pass in an upward and downward direction.
The example is inspired by / based on:
- [Tensorflow 2 Object Counting](https://github.com/TannerGilbert/Tensorflow-2-Object-Counting)
- [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)
- [tensorflow_object_counting_api](https://github.com/ahmetozlu/tensorflow_object_counting_api)

## Demo

![cumulative object counting](media/cumulative-object-counting.gif)

## Installation

Running this example requires a **Luxonis OAK2 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-m MODEL, --model MODEL
                      HubAI reference of the object detection model. (default: luxonis/mobilenet-ssd:300x300)
-d DEVICE, --device DEVICE
                      Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                      FPS limit for the model runtime. (default: None)
-media MEDIA_PATH, --media_path MEDIA_PATH
                      Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-a AXIS, --axis AXIS
                      Axis for cumulative counting (either x or y). (default: x)
-roi ROI_POSITION, --roi_position ROI_POSITION
                      osition of the axis (if 0.5, axis is placed in the middle of the frame). (default: 0.5)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the experiment using `mobilenet-ssd` model counting for people on your camera input.

```bash
python3 main.py \
    --model <HubAI model reference> \
    --media_path <path to media file> \
    --axis y
```

This will run the experiment using a custom model counting for a custom object class (the experiment is hardcoded to use the model class with the smallest ID) on a media file input.
The objects are counted when crossing the y-axis positioned in the middle of the frame (default position).

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
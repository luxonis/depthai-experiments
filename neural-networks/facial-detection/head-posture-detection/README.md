# Head posture detection

This example demonstrates calculating head poses in a video stream. This is achieved by using a two stage inference DepthAI pipeline to first detect faces and then infer the pitch, yaw and roll of each one. The demo uses [YuNet Face detection model](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page) for the first stage and to infer the pose it uses [Head posture model](https://hub.luxonis.com/ai/models/068ac18a-de71-4a6e-9f0f-42776c0ef980?view=page) from the Luxonis Hub.

## Demo

<!-- ![Head pose estimation](https://user-images.githubusercontent.com/18037362/172148301-45adb7ce-3aab-478f-8cad-0c05f349ce50.gif) -->

![Head pose estimation](media/head_pose.gif)

### How it works

1. Color camera produces high-res frames, that are sent to Face detection NN node.
1. The object detections (faces) are processed by CropConfigsCreator node which generates a dai.ImageManipConfigV2 for each detection.
1. ImageManipV2 node uses the generated configs to crop all detections from the original image frame. It also resizes the cropped detections to (60, 60) which is the input size of the Head posture NN.
1. Head posture NN node infers yaw, pitch and roll from the outputs of the ImageManipV2 node.
1. The original frame, detections and the head postures get synced back together with the use of two Gather nodes (first sync detections and postures, then sync all of them at once to the frame)
1. The synced data is then sent to an AnnotationNode to generate text and bounding box.
1. The annotations and frame are then registered to a dai.RemoteConnection() object to display them in a local browser.

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30.0 for RVC4, 6.0 for RVC2)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the experiment on the default connected device.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the experiment with the default device and the video file.

```bash
python3 main.py -d <DEVICE-IP / DEVICE-MXID >
```

This runs the experiment with a specific device.

### Standalone Mode

Running the experiment in the [Standalone mode](https://rvc4.docs.luxonis.com/software/depthai/standalone/) runs the app entirely on the device.
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

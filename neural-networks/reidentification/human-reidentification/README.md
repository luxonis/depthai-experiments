# Human Re-Identification

This experiment demonstrates how to build a 2-stage DepthAI pipeline for human pose / human face reidentification.
The pipeline consists of a detection model (
[SCRFD Pose](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c?view=page),
[SCRFD Face](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page), or
[YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page)
) predicting bounding boxes around the relevant objects, and a recognition model (
[OSNet](https://hub.luxonis.com/ai/models/6d853621-818b-4fa4-bd9a-d9bdcb5616e6?view=page) or
[ArcFace](https://hub.luxonis.com/ai/models/e24a577e-e2ff-4e4f-96b7-4afb63155eac?view=page)
) providing embeddings for each of the detected objects.
Object reidentification is achieved by calculating cosine similarity between the embeddings.

## Demo

[![human pose reidentification](media/human_pose_reidentification.gif)](media/human_pose_reidentification.gif)
[![human face reidentification](media/human_face_reidentification.gif)](media/human_face_reidentification.gif)

<sup>[Source](https://www.pexels.com/video/happy-people-walking-on-green-grass-7551577/)</sup>

## Installation

Running this example requires a **Luxonis OAK device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

Here is a list of all available parameters:

```
-id IDENTIFY, --identify IDENTIFY
                    Determines what object to use for identification ('pose' or 'face'). (default: 'pose')
-cos COS_SIMILARITY_THRESHOLD, --cos_similarity_threshold COS_SIMILARITY_THRESHOLD
                    Cosine similarity between object embeddings above which detections are considered as belonging to the same object. (default: 0.5)
-media MEDIA_PATH, --media_path MEDIA_PATH
                    Path to the media file you aim to run the model on. If not set, the model will run on the camera input. (default: None)
-fps FPS_LIMIT, --fps_limit FPS_LIMIT
                    FPS limit for the model runtime. (default: 30.0)
-d DEVICE, --device DEVICE
                    Optional name, DeviceID or IP of the camera to connect to. (default: None)
```

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app. Below are some examples of how to run the example.

#### Examples

```bash
python3 main.py
```

This will run the human pose reidentification with the default device and camera input and a cosine similarity threshold of 0.8 (default for the 'pose' option).

```bash
python3 main.py \
    -id face \
    -fps 5
```

This will run the human face reidentification with the default device and camera input at 5 FPS and a cosine similarity threshold of 0.1 (default for the 'face' option).

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

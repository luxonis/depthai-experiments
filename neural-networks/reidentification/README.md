# Overview

This experiment demonstrates how to build a 2-stage DepthAI pipeline for human pose / human face re-identification.
The pipeline consists of a detection model (
[SCRFD Pose](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c?view=page),
[SCRFD Face](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page), or
[YuNet](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page)
) predicting bounding boxes around the relevant objects, and a recognition model (
[OSNet](https://hub.luxonis.com/ai/models/6d853621-818b-4fa4-bd9a-d9bdcb5616e6?view=page) or
[ArcFace](https://hub.luxonis.com/ai/models/e24a577e-e2ff-4e4f-96b7-4afb63155eac?view=page)
) providing embeddings for each of the detected objects.
Object re-identification is achieved by calculating cosine similarity between the embeddings.

**WARNING:** The experiment currently works only on RVC4 devices.

## Demo

[![human pose re-identification](media/human_pose_reidentification.gif)](media/human_pose_reidentification.gif)
[![human face re-identification](media/human_face_reidentification.gif)](media/human_face_reidentification.gif)

<sup>[Source](https://www.pexels.com/video/happy-people-walking-on-green-grass-7551577/)</sup>

## Installation

Running this example requires a **Luxonis OAK4 device** connected to your computer. You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py -det <DET_MODEL> -rec <REC_MODEL> -cos <CSIM> -media <MEDIA> -fps <FPS_LIMIT> --device <DEVICE>
```

- `<DET_MODEL>`: Detection model reference from Luxonis HubAI.
- `<REC_MODEL>`: Recognition model reference from Luxonis HubAI.
- `<CSIM>` \[OPTIONAL\]: Cosine similarity between object embeddings above which detections are considered as belonging to the same object. Default: `0.5`.
- `<DEVICE>` \[OPTIONAL\]: Device IP or ID. Default: `None` - use the first identified device connected to the host.
- `<MEDIA>` \[OPTIONAL\]: Path to the video file. Default: `None` - use camera input.
- `<FPS_LIMIT>` \[OPTIONAL\]: Limit of the video/camera FPS. Beware that if providing a video file with higher FPS, a slowed-down video will be shown (and vice-versa if providing a video file with higher FPS). Default: `30`.

#### Examples

```bash
python3 main.py \
    -det luxonis/scrfd-person-detection:25g-640x640 \
    -rec luxonis/osnet:imagenet-128x256 \
    -cos 0.8 \
    -fps 5
```

This will run the human pose re-identification with the default device and camera input at 5 FPS and a cosine similarity threshold of 0.8.

```bash
python3 main.py \
    -det luxonis/scrfd-face-detection:10g-640x640 \
    -rec luxonis/arcface:lfw-112x112 \
    -media <path/to/video.mp4> \
    -fps 5
```

This will run the human face re-identification with the default device and video input at 5 FPS.

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

This will run the human pose re-identification with the default device and camera input at 5 FPS and a cosine similarity threshold of 0.8.

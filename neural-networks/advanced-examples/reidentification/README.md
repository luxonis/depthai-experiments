# Overview
This experiment demonstrates how to build a 2-stage DepthAI pipeline for human pose/face re-identification. 
The pipeline consists of a detector (
    [SCRFD Person detection](https://hub.luxonis.com/ai/models/c3830468-3178-4de6-bc09-0543bbe28b1c?view=page)/
    [SCRFD Face detection](https://hub.luxonis.com/ai/models/1f3d7546-66e4-43a8-8724-2fa27df1096f?view=page)/
    [YuNet face detection](https://hub.luxonis.com/ai/models/5d635f3c-45c0-41d2-8800-7ca3681b1915?view=page)
) and a recognition (
    [OSNet](https://hub.luxonis.com/ai/models/6d853621-818b-4fa4-bd9a-d9bdcb5616e6?view=page)/
    [ArcFace](https://hub.luxonis.com/ai/models/e24a577e-e2ff-4e4f-96b7-4afb63155eac?view=page)
) model. 
The experiment is currently running only on RVC4.

## Demo

[![pose re-identification](media/pose_reidentification.gif)](media/pose_reidentification.gif)
[![face re-identification](media/face_reidentification.gif)](media/face_reidentification.gif)

[SOURCE](https://www.pexels.com/video/happy-people-walking-on-green-grass-7551577/) video by RDNE Stock project from Pexels.

## Installation

You need to prepare a Python environment with [DepthAI](https://pypi.org/project/depthai/) and [DepthAI Nodes](https://pypi.org/project/depthai-nodes/) packages installed. You can do this by running:

```bash
pip install -r requirements.txt
```

**NOTE: depthai_dev and depthai_nodes main - they are not on PyPi yet, so you need to install them from GitHub.**

## Usage

You can run the experiment fully on device (`STANDALONE` mode) or using your your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --det_model <DET_MODEL> --rec_model <REC_MODEL> --cos_similarity_threshold <CSIM> --media_path <MEDIA> --fps_limit <FPS_LIMIT> --device <DEVICE> 
```

- `<DET_MODEL>`: Detection model reference from Luxonis HubAI.
- `<REC_MODEL>`: Recognition model reference from Luxonis HubAI.
- `<CSIM>` [OPTIONAL]: Cosine similarity above which detections are considered as belonging to the same object.. Default: `0.5`.
- `<DEVICE>` [OPTIONAL]: Device IP or ID. Default: `None` - device connected to the host.
- `<MEDIA>` [OPTIONAL]: Path to the video file. Default `None` - camera input.
- `<FPS_LIMIT>` [OPTIONAL]: Limit of the camera/media FPS. Beware that if providing a video file with higher FPS, a slowed-down video will be shown (and vice-versa if the FPS of the provided media is lower). Default: `30`.

#### Examples

```bash
python3 main.py
```

This will run the human pose estimation experiment with the default device, default model, and camera input.

```bash
python3 main.py --media <PATH_TO_VIDEO>
```

This will run the human pose estimation experiment with the default device and the video file.

```bash
python3 main.py --model luxonis/lite-hrnet:30-coco-192x256 --fps_limit 5
```

This will run the human pose estimation experiment with the default device and camera input, but with the `luxonis/lite-hrnet:30-coco-192x256` model and a `5` FPS limit.


### How it works

1. Color camera produces high-res frames, sends them to host, Script node and downscale ImageManip node
2. Downscale ImageManip will downscale from high-res frame to 544x320, required by 1st NN in this pipeline; object detection model [person-detection-retail-0013 model](https://docs.openvino.ai/latest/omz_models_model_person_detection_retail_0013.html)
3. 544x320 frames are sent from downscale ImageManip node to the object detection model (MobileNetSpatialDetectionNetwork)
4. Object detections are sent to the Script node
5. Script node first syncs object detections msg with frame. It then goes through all detections and creates ImageManipConfig for each detected person. These configs then get sent to ImageManip together with synced high-res frame
6. ImageManip will crop only the person out of the original frame. It will also resize the person frame to required size (128,256) by the person-reidentification NN model
7. Person frames get sent to the 2nd NN - [person-reidentification](https://docs.openvino.ai/latest/omz_models_model_person_reidentification_retail_0288.html) NN model. NN results are sent back to the host
8. Frames, object detections, and reidentification results are all **synced on the host** side and then displayed to the user
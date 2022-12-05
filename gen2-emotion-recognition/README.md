# Gen2 Emotion recognition

This demo recognizes facial emotions (`neutral`, `happy`, `sad`, `surprise`, `anger`). Demo uses [face-detection-retail-0004](https://docs.openvino.ai/2021.4/omz_models_model_face_detection_retail_0004.html) model to detect faces, crops them on the device using Script node, and then sends face frames to [emotions-recognition-retail-0003](https://docs.openvino.ai/2021.4/omz_models_model_emotions_recognition_retail_0003.html) model which estimates emotions.

## Demo

![Demo](https://user-images.githubusercontent.com/18037362/159129815-f41b2863-67c4-4e6c-a1b5-54a78cc6b8a8.png)

### How it works

1. Color camera produces high-res frames, sends them to host, Script node and downscale ImageManip node
2. Downscale ImageManip will downscale from high-res frame to 300x300, required by 1st NN in this pipeline; object detection model
3. 300x300 frames are sent from downscale ImageManip node to the object detection model (MobileNetSpatialDetectionNetwork)
4. Object detections are sent to the Script node
5. Script node first syncs object detections msg with frame. It then goes through all detections and creates ImageManipConfig for each detected face. These configs then get sent to ImageManip together with synced high-res frame
6. ImageManip will crop only the face out of the original frame. It will also resize the face frame to required size (64,64) by the emotions recognition NN model
7. Face frames get send to the 2nd NN - emotions NN model. NN recognition results are sent back to the host
8. Frames, object detections, and recognition results are all **synced on the host** side and then displayed to the user

## 2-stage NN pipeline graph

![image](https://user-images.githubusercontent.com/18037362/179375207-1ccf27a6-59bb-4a42-8cae-d8908c4ed51a.png)

[DepthAI Pipeline Graph](https://github.com/geaxgx/depthai_pipeline_graph#depthai-pipeline-graph-experimental) was used to generate this image.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
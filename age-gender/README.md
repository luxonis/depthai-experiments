[中文文档](README.zh-CN.md)

# Age & Gender recognition

This example demonstrates how to run 2 stage inference with DepthAI library.
It recognizes age and gender of all detected faces on the frame. Demo uses [face-detection-retail-0004](https://docs.openvino.ai/2021.4/omz_models_model_face_detection_retail_0004.html) model to detect faces, crops them on the device using Script node, and then sends face frames to [age-gender-recognition-retail-0013](https://docs.openvino.ai/latest/omz_models_model_age_gender_recognition_retail_0013.html) model which estimates age and gender of the face.

## Demo

[![Age & Gender recognition](https://user-images.githubusercontent.com/18037362/159127397-f75a96a9-f699-4bc8-bb54-39f998c044be.png)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

### How it works

1. Color camera produces high-res frames, sends them to host, Script node and downscale ImageManip node
2. Downscale ImageManip will downscale from high-res frame to 300x300, required by 1st NN in this pipeline; object detection model
3. 300x300 frames are sent from downscale ImageManip node to the object detection model (MobileNetSpatialDetectionNetwork)
4. Object detections are sent to the Script node
5. Script node first syncs object detections msg with frame. It then goes through all detections and creates ImageManipConfig for each detected face. These configs then get sent to ImageManip together with synced high-res frame
6. ImageManip will crop only the face out of the original frame. It will also resize the face frame to required size (62,62) by the age-gender recognition NN model
7. Face frames get send to the 2nd NN - age-gender NN model. NN recognition results are sent back to the host
8. Frames, object detections, and recognition results are all **synced on the host** side and then displayed to the user

## 2-stage NN pipeline graph

![image](https://user-images.githubusercontent.com/18037362/179375207-1ccf27a6-59bb-4a42-8cae-d8908c4ed51a.png)

[DepthAI Pipeline Graph](https://github.com/geaxgx/depthai_pipeline_graph#depthai-pipeline-graph-experimental) was used to generate this image.

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

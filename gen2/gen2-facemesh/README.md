## Facial Landmarks on DepthAI

This example shows an implementation of Facial Landmark detection that's used in [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html). The TFLite model was taken from [MediaPipe](https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark) and converted to blob so that it can run on OAK devices.

Model takes input image of size 192 x 192, and predicts 468 facial landmarks and a score.

## Demo

https://github.com/luxonis/depthai-experiments/assets/18037362/04d594e2-2238-49d6-a818-a9079270beff


## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```

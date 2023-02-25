# Object tracking using DeepSORT

This example demonstrates how to run 2 stage object tracking using [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) with DepthAI library.
It tracks detected objects in the frames. Demo uses [YoloV6n](https://github.com/meituan/YOLOv6) model to detect objects, crops them on the device using Script node, and then sends object frames to [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) feature extraction model which computes the embedding of the object.

## Demo

![Object tracking using DeepSORT Demo](assets/deepsort-tracking-demo.png)

### How it works

1. Color camera produces high-res frames, sends them to host, Script node and downscale ImageManip node
2. Downscale ImageManip will downscale from high-res frame to 640x640, required by 1st NN in this pipeline; object detection model
3. 640x640 frames are sent from downscale ImageManip node to the object detection model (YoloSpatialDetectionNetwork/YoloDetectionNetwork)
4. Object detections are sent to the Script node
5. Script node first syncs object detections msg with frame. It then goes through all detections and creates ImageManipConfig for each detected object. These configs then get sent to ImageManip together with synced high-res frame
6. ImageManip will crop only the object out of the original frame. It will also resize the object frame to required size (224,224) by the feature extraction NN model
7. Object frames get send to the 2nd NN - feature extraction NN model. NN embedding results are sent back to the host
8. Frames, object detections, and embedding results are all **synced on the host** side, the detections and embeddings are passed to the [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) tracker to update its state and then the tracked objects are displayed to the user

## 2-stage NN pipeline graph

![Object tracking using DeepSORT](assets/deepsort-tracking-pipeline.png)

[DepthAI Pipeline Graph](https://github.com/geaxgx/depthai_pipeline_graph#depthai-pipeline-graph-experimental) was used to generate this image.


## Usage

### Navigate to directory

```
cd ./api
```

### Pre-requisites

```
python3 -m pip install -r requirements.txt
```

### Launch the script

```
python3 main.py
```

## Credits
In this project we have used [YoloV6n](https://github.com/meituan/YOLOv6) for object detection, [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) as a feature extractor, and [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) for tracking.
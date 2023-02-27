[中文文档](README.zh-CN.md)

Face recognition
================

This example demonstrates the DepthAI running [face detection network](https://docs.openvinotoolkit.org/2021.3/omz_models_model_face_detection_retail_0004.html), [head posture estimation network](https://docs.openvinotoolkit.org/2021.3/omz_models_model_head_pose_estimation_adas_0001.html) and [face recognition network](https://docs.openvinotoolkit.org/2021.3/omz_models_model_face_recognition_mobilefacenet_arcface.html). It recognizes multiple faces at once on the frame.

## Demo

[![Face recognition](https://user-images.githubusercontent.com/18037362/159522552-fde15cd4-4343-492e-be44-ae07f06c1d2e.gif)](https://youtu.be/Xb1cXu_SIbo)


### How it works

1. The color camera produces high-res frames, sends them to host, Script node, and downscale ImageManip node.
2. Downscale ImageManip will downscale from high-res frame to `300x300`, required by 1st NN in this pipeline; object detection model.
3. `300x300` frames are sent from downscale ImageManip node to the object detection model (MobileNetDetectionNetwork).
4. Object detections are sent to the Script node.
5. Script node first syncs object detections msg with frame. It then goes through all detections and creates ImageManipConfig for each detected face. These configs then get sent to ImageManip together with synced high-res frame.
6. ImageManip will crop only the face out of the original frame. It will also resize the face frame to the required size (`60x60`) by the head pose estimation NN model.
7. Face frames get sent to the 2nd NN - head pose estimation NN model. NN estimations results are sent back to the Script node together with the passthrough frame (for syncing).
8. Script node syncs the head pose estimation, high-res frame, and face detection results. It then creates ImageManipConfig that will rotate the bounding box so that the face will also be vertical, which significantly improves face recognition accuracy.
9. Created ImageManipConfig and high-res frame get sent to another ImageManip node, which crops rotated rectangle and feeds the `112x112` frame to the 3rd NN: face recognition the arcface model.
10. Frames, object detections, and recognition results are all **synced on the host** side.
11. Face recognition results are matched with faces in the database using cosine distance (inside `FaceRecognition` class) and then displayed to the user.

## Pipeline graph

![image](https://user-images.githubusercontent.com/18037362/179375078-c2544a58-a9b3-464f-9f80-2e7deb49a727.png)

[DepthAI Pipeline Graph](https://github.com/geaxgx/depthai_pipeline_graph#depthai-pipeline-graph-experimental) was used to generate this image.


## Potential improvements

This is a demo app showcasing the capability of depthai, not a production-ready solution. A few potential improvements:

- Use scalable DB system for comparing face embeddings. If you have a large DB (eg. thousands of people), this will be extremely important, as comparing numpy arrays in python `for` loop isn't very fast/efficient.
- Trying a different face recognition model that could potentially be more accurate. A few other pretrained models: [Sphereface](https://docs.openvino.ai/2021.4/omz_models_model_Sphereface.html), [face_recognition_resnet100_arcface_onnx](https://docs.openvino.ai/2021.4/omz_models_model_face_recognition_resnet100_arcface_onnx.html), [face_reidentification_retail_0095](https://docs.openvino.ai/2021.4/omz_models_model_face_reidentification_retail_0095.html), [facenet-20180408-102900](https://docs.openvino.ai/2021.4/omz_models_model_facenet_20180408_102900.html).
- Use a different (potentially more accurate) comparison technique to compare face embeddings. We currently use cosine distance.
- Use depth as well to increase face recognition accuracy. We haven't found any opensource/pretrained NN models that would accept color+depth frames for better accuracy, but Varun from LearnOpenCv used depth in the [anti-spoofing face recognition system](https://learnopencv.com/anti-spoofing-face-recognition-system-using-oak-d-and-depthai/) to determine whether the image is flat (2D image, eg. from aphone).
- Use anti-spoofing model, an example would be [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) repo.

## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

```bash
usage: main.py [-name NAME]

optional arguments:
  -name, --name     Name of the person for database saving [Optional]
```

**Before this example works, you have to "teach" it what face to associate with which name:**

1. Run `python3 main.py --name JohnDoe`. Then you should face the camera to JohnDoe from different angles, so he will later be recognized from different angles as well. This will just save (his) face vectors to the person's databse (in this case `JohnDoe.npz`).
2. Repeat step 1 for other people you would like to recognize
3. Run `python3 main.py` for face recognition demo. Whenever the device sees a new face, it will calculate the face vector (arcface NN model) and it will get compared with other vectors from the databases (`.npz`) using cosine distance.


## How it works:

### 1. Run the face detection model

> Run the [face-detection-retail-0004](models/face-detection-retail-0004_openvino_2020_1_4shave.blob) model to 
> detect the face in the image and intercept the facial image.
> 
> ![detection_face](assets/detection_face.png)

### 2. Run head-pose-estimation model

> Run the [head-pose-estimation-adas-0001](models/head-pose-estimation-adas-0001.blob) model to 
> Detect head tilt angle and adjust head posture.
> 
>![face_corr](assets/face_corr.png)

### 3. Run face recognition model

> Run the [face-recognition-mobilefacenet-arcface.blob](models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob) model to 
> Recognize the face.
>
> ![face_reg](assets/face_reg.png)

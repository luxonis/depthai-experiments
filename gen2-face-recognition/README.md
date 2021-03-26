Face recognition
================

This example demonstrates the Gen2 Pipeline Builder running [face detection network](https://docs.openvinotoolkit.org/latest/omz_models_model_face_detection_retail_0004.html)  ,[head posture estimation network](https://docs.openvinotoolkit.org/latest/omz_models_model_head_pose_estimation_adas_0001.html) and [face recognition network](https://docs.openvinotoolkit.org/latest/omz_models_model_face_recognition_mobilefacenet_arcface.html)

## How it works:

### 1. Run the face detection model

> Run the [face-detection-retail-0004](models/face-detection-retail-0004_openvino_2020_1_4shave.blob) model to 
> detect the face in the image and intercept the facial image.
> 
> ![detection_face](images/detection_face.png)

### 2. Run head-pose-estimation model

> Run the [head-pose-estimation-adas-0001](models/head-pose-estimation-adas-0001.blob) model to 
> Detect head tilt angle and adjust head posture.
> 
>![face_corr](images/face_corr.png)

### 3. Run face recognition model

> Run the [face-recognition-mobilefacenet-arcface.blob](models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob) model to 
> Recognize the face.
>
> ![face_reg](images/face_reg.png)

--------------------

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```bash
   python3 -m pip install -r requirements.txt
   ```


## Usage

```bash
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        The path of the video file used for inference (conflicts with -cam)
  -db, --databases      Save data (only used when running recognition network)
  -n NAME, --name NAME  Data name (used with -db) [Optional]

```

To use with video files and build a face database
```bash
python main.py -db -n <name> -vid <path>
```

To use with DepthAI 4K RGB camera, use instead
```bash
python main.py -db -n <name> -cam
```

To use with a video file, run the script with the following arguments

```bash
python main.py -vid <path>
```

To use with DepthAI 4K RGB camera, use instead
```bash
python main.py -cam
```

> Press 'q' to exit the program.

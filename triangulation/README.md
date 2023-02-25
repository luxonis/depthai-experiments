# Triangulation - Stereo neural inference demo

Because there are often application-specific host-side filtering to be done on the stereo
neural inference results, and because these calculations are lightweight
(i.e. could be done on an ESP32), we leave the triangulation itself to the host.

This 3D visualizer is for the facial landmarks demo, and uses OpenGL and OpenCV. Consider it a draft/reference at this point.

Demo uses 2-stage inferencing; 1st NN model is [face-detection-retail-0004](https://docs.openvino.ai/2021.4/omz_models_model_face_detection_retail_0004.html) and 2nd NN model is [landmarks-regression-retail-0009](https://docs.openvino.ai/2021.4/omz_models_model_landmarks_regression_retail_0009.html).

> This demo is only available for DepthAI API.

## Demo

![Stereo Inference GIF](https://user-images.githubusercontent.com/59799831/132098832-70a2d0b9-1a30-4994-8dad-dc880a803fb3.gif)

## Installation

```
cd ./api
sudo apt-get install python3-pygame
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

You should see 5 windows appear:
- `mono_left` which will show camera output from left mono camera + face bounding box & facial landmarks
- `mono_right` which will show camera output from right mono camera + face bounding box & facial landmarks
- `crop_left` which will show 48x48 left cropped image that goes into the second NN + facial landmarsk that get outputed from the second NN
- `crop_right` which will show 48x48 right cropped image that goes into the second NN + facial landmarsk that get outputed from the second NN
- `pygame window` which will show the triangulation results

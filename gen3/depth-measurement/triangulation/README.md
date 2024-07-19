# Triangulation - Stereo neural inference demo

Because there are often application-specific host-side filtering to be done on the stereo
neural inference results, and because these calculations are lightweight
(i.e. could be done on an ESP32), we leave the triangulation itself to the host.

Demo uses 2-stage inferencing; 1st NN model is [face-detection-retail-0004](https://docs.openvino.ai/2021.4/omz_models_model_face_detection_retail_0004.html) and 2nd NN model is [landmarks-regression-retail-0009](https://docs.openvino.ai/2021.4/omz_models_model_landmarks_regression_retail_0009.html).

## Demo

![Stereo Inference GIF](https://user-images.githubusercontent.com/59799831/132098832-70a2d0b9-1a30-4994-8dad-dc880a803fb3.gif)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

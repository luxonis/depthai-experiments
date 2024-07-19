# Age & Gender recognition

This example demonstrates how to run 2 stage inference with DepthAI library.
It recognizes age and gender of all detected faces on the frame. Demo uses [face-detection-retail-0004](https://docs.openvino.ai/2021.4/omz_models_model_face_detection_retail_0004.html) model to detect faces, crops them on the device using Script node, and then sends face frames to [age-gender-recognition-retail-0013](https://docs.openvino.ai/latest/omz_models_model_age_gender_recognition_retail_0013.html) model which estimates age and gender of the face.

## Demo

[![Gen2 Age & Gender recognition](https://user-images.githubusercontent.com/18037362/159127397-f75a96a9-f699-4bc8-bb54-39f998c044be.png)](https://www.youtube.com/watch?v=PwnVrPaF-vs "Age/Gender recognition on DepthAI")

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
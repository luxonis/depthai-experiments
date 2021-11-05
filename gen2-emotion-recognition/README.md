# Gen2 Emotion recognition

This demo recognizes facial emotions (`neutral`, `happy`, `sad`, `surprise`, `anger`). Demo uses [face-detection-retail-0004](https://docs.openvino.ai/2021.4/omz_models_model_face_detection_retail_0004.html) model to detect faces, crops them on the device using Script node, and then sends face frames to [emotions-recognition-retail-0003](https://docs.openvino.ai/2021.4/omz_models_model_emotions_recognition_retail_0003.html) model which estimates emotions.

## Demo

![Demo](https://user-images.githubusercontent.com/18037362/140508779-f9b1465a-8bc1-48e0-8747-80cdb7f2e4fc.png)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
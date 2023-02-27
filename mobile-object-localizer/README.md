##  Mobile Object Localizer on DepthAI

This example shows an implementation of Mobile Object Localizer from [Tensorflow Hub](https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1).

Blob is taken from [Pinto Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer).

Input video is resized to 192 x 192 (W x H). Output is a list of 100 scores and bounding boxes (see implementation for details).

![Image example](https://user-images.githubusercontent.com/18037362/140496684-e886fc00-612d-44dd-a6fe-c0d47988246f.gif)
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
# DepthAI API
python3 main.py [-t THRESHOLD]  # default threshold is 0.2

# DepthAI SDK
python3 main.py
```
